"""Build a labeled feature table from dispositioned TOIs."""

from __future__ import annotations

import logging
import multiprocessing as mp
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from multiprocessing.connection import wait
from pathlib import Path

import pandas as pd

from exovet.data.lightcurves import DEFAULT_AUTHORS, load_detrended
from exovet.data.toi import row_to_candidate
from exovet.features import catalog_features, compute_features

log = logging.getLogger(__name__)

DEFAULT_DATASET = Path("data/features.csv")
DEFAULT_CANDIDATES = Path("data/candidates.csv")
TARGET_TIMEOUT = 600


def _features_for(row: pd.Series, authors: Sequence[str]) -> dict | None:
    cand = row_to_candidate(row)
    if cand is None:
        return None
    try:
        lc = load_detrended(cand, authors=authors)
        features = compute_features(lc.time, lc.flux, cand, lc.centroids)
    except Exception as exc:  # noqa: BLE001 - network errors, missing data, corrupt files
        log.warning("Skipping %s: %s", cand.name, exc)
        return None
    label = row.get("label")
    record = {"toi": cand.toi, "tic_id": cand.tic_id, "author": lc.author}
    if pd.notna(label):
        record["label"] = int(label)
    return {**record, **features}


def _run_target(conn, target, row, authors) -> None:
    conn.send(target(row, authors))
    conn.close()


def _iter_records(
    rows: Iterable[pd.Series],
    authors: Sequence[str],
    workers: int,
    timeout: float,
    target: Callable[[pd.Series, str], dict | None] = _features_for,
) -> Iterator[dict | None]:
    """Yield ``target(row, authors)`` for each row, in completion order.

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
            proc = ctx.Process(target=_run_target, args=(send, target, row, authors), daemon=True)
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


def refresh_catalog_features(dataset: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    """Recompute labels and catalog-only features for existing rows.

    Needs no light curves, so new catalog features (or catalog updates) reach
    the whole dataset without re-downloading anything. Rows whose TOI has left
    the catalog, lost its label, or no longer converts are kept unchanged.
    """
    by_toi = catalog.set_index("TOI")
    updates = {}
    for toi in dataset["toi"]:
        if toi not in by_toi.index:
            continue
        row = by_toi.loc[toi].copy()  # TOI numbers are unique in the catalog
        row["TOI"] = toi
        cand = row_to_candidate(row)
        if cand is None:
            continue
        label = {"label": int(row["label"])} if pd.notna(row["label"]) else {}
        updates[toi] = {**label, **catalog_features(cand)}
    if not updates:
        return dataset

    fresh = pd.DataFrame.from_dict(updates, orient="index")
    out = dataset.set_index("toi")
    for column in fresh.columns:
        if column not in out.columns:
            out[column] = float("nan")
    out.loc[fresh.index, fresh.columns] = fresh  # unlike update(), also copies NaN
    if "label" in out and out["label"].notna().all():
        out["label"] = out["label"].astype(int)
    # Identifiers first, then catalog features, matching compute_features' order.
    base = [c for c in ("tic_id", "author", "label") if c in out.columns]
    catalog_cols = [c for c in fresh.columns if c not in base]
    rest = [c for c in out.columns if c not in base and c not in catalog_cols]
    return out[base + catalog_cols + rest].reset_index()


def build_dataset(
    catalog: pd.DataFrame,
    out: Path = DEFAULT_DATASET,
    limit: int | None = None,
    authors: Sequence[str] = DEFAULT_AUTHORS,
    workers: int = 4,
    seed: int = 0,
    timeout: float = TARGET_TIMEOUT,
    labeled: bool = True,
    detection: str | None = None,
) -> pd.DataFrame:
    """Compute features for TOIs, appending to ``out`` as it goes.

    With ``labeled`` the dispositioned TOIs are used, otherwise the unresolved
    ones (PC/APC and undispositioned), whose rows carry no label and are meant
    for scoring rather than training. ``detection`` keeps only TOIs whose
    catalog Detection field mentions that pipeline, e.g. SPOC for the targets
    that have 2-minute light curves.

    TOIs are visited in a seeded random order so a ``limit`` gives a
    representative sample rather than the (brighter, better-observed) earliest
    TOIs. Rows already present in ``out`` are skipped, so an interrupted run
    resumes; their labels and catalog-only features are refreshed first. A target still running after ``timeout`` seconds is killed and
    skipped.
    """
    out = Path(out)
    done: set[str] = set()
    columns = None
    if out.exists():
        existing = refresh_catalog_features(pd.read_csv(out, dtype={"toi": str}), catalog)
        existing.to_csv(out, index=False)
        done = set(existing["toi"])
        columns = list(existing.columns)

    has_label = catalog["label"].notna()
    targets = catalog[has_label if labeled else ~has_label]
    if detection:
        targets = targets[targets["Detection"].str.contains(detection, na=False)]
    targets = targets.sample(frac=1, random_state=seed)
    if limit is not None:
        targets = targets.head(limit)
    targets = targets[~targets["TOI"].isin(done)]

    out.parent.mkdir(parents=True, exist_ok=True)
    records = _iter_records((row for _, row in targets.iterrows()), authors, workers, timeout)
    for i, record in enumerate(records, 1):
        if record is None:
            continue
        row = pd.DataFrame([record])
        if columns is None:
            columns = list(row.columns)
        # reindex, not selection: a resumed file may carry columns this row lacks.
        row.reindex(columns=columns).to_csv(out, mode="a", header=not out.exists(), index=False)
        log.info("[%d/%d] TOI-%s", i, len(targets), record["toi"])

    if not out.exists():  # every target failed, e.g. the archive is down
        return pd.DataFrame()
    return pd.read_csv(out, dtype={"toi": str})
