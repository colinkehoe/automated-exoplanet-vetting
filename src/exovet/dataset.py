"""Build a labeled feature table from dispositioned TOIs."""

from __future__ import annotations

import logging
import multiprocessing as mp
import time
from collections.abc import Callable, Iterable, Iterator
from multiprocessing.connection import wait
from pathlib import Path

import pandas as pd

from exovet.data.lightcurves import load_detrended
from exovet.data.toi import row_to_candidate
from exovet.features import compute_features

log = logging.getLogger(__name__)

DEFAULT_DATASET = Path("data/features.csv")
TARGET_TIMEOUT = 600


def _features_for(row: pd.Series, author: str) -> dict | None:
    cand = row_to_candidate(row)
    if cand is None:
        return None
    try:
        t, flux, _ = load_detrended(cand, author=author)
        features = compute_features(t, flux, cand)
    except Exception as exc:  # noqa: BLE001 - network errors, missing data, corrupt files
        log.warning("Skipping %s: %s", cand.name, exc)
        return None
    return {"toi": cand.toi, "tic_id": cand.tic_id, "label": int(row["label"]), **features}


def _run_target(conn, target, row, author) -> None:
    conn.send(target(row, author))
    conn.close()


def _iter_records(
    rows: Iterable[pd.Series],
    author: str,
    workers: int,
    timeout: float,
    target: Callable[[pd.Series, str], dict | None] = _features_for,
) -> Iterator[dict | None]:
    """Yield ``target(row, author)`` for each row, in completion order.

    Each row runs in its own process so the parent can kill it after
    ``timeout`` seconds. MAST downloads can stall in ways no in-process
    timeout reliably interrupts (astroquery disables socket timeouts, polls
    without a deadline, and swallows exceptions). Processes rather than
    threads also because lightkurve/astropy FITS reading is not thread-safe.
    A killed or crashed target yields None.
    """
    ctx = mp.get_context("spawn")
    pending = list(rows)[::-1]
    running = {}  # receiving connection -> (process, TOI, start time)
    while pending or running:
        while pending and len(running) < workers:
            row = pending.pop()
            recv, send = ctx.Pipe(duplex=False)
            proc = ctx.Process(target=_run_target, args=(send, target, row, author), daemon=True)
            proc.start()
            send.close()
            running[recv] = (proc, row["TOI"], time.monotonic())

        wait(list(running), timeout=1)
        for recv, (proc, toi, started) in list(running.items()):
            record = None
            if recv.poll():  # a result, or EOF if the process died without one
                try:
                    record = recv.recv()
                except EOFError:
                    log.warning("Skipping TOI-%s: worker exited with code %s", toi, proc.exitcode)
            elif time.monotonic() - started > timeout:
                proc.kill()
                log.warning("Skipping TOI-%s: killed after %d s", toi, timeout)
            else:
                continue
            proc.join()
            recv.close()
            del running[recv]
            yield record


def build_dataset(
    catalog: pd.DataFrame,
    out: Path = DEFAULT_DATASET,
    limit: int | None = None,
    author: str = "SPOC",
    workers: int = 4,
    seed: int = 0,
    timeout: float = TARGET_TIMEOUT,
) -> pd.DataFrame:
    """Compute features for labeled TOIs, appending to ``out`` as it goes.

    TOIs are visited in a seeded random order so a ``limit`` gives a
    representative sample rather than the (brighter, better-observed) earliest
    TOIs. Rows already present in ``out`` are skipped, so an interrupted run
    resumes. A target still running after ``timeout`` seconds is killed and
    skipped.
    """
    out = Path(out)
    done: set[str] = set()
    if out.exists():
        done = set(pd.read_csv(out, dtype={"toi": str})["toi"])

    labeled = catalog[catalog["label"].notna()].sample(frac=1, random_state=seed)
    if limit is not None:
        labeled = labeled.head(limit)
    labeled = labeled[~labeled["TOI"].isin(done)]

    out.parent.mkdir(parents=True, exist_ok=True)
    records = _iter_records((row for _, row in labeled.iterrows()), author, workers, timeout)
    for i, record in enumerate(records, 1):
        if record is None:
            continue
        pd.DataFrame([record]).to_csv(out, mode="a", header=not out.exists(), index=False)
        log.info("[%d/%d] TOI-%s", i, len(labeled), record["toi"])

    return pd.read_csv(out, dtype={"toi": str})
