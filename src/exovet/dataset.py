"""Build a labeled feature table from dispositioned TOIs."""

from __future__ import annotations

import logging
import signal
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from exovet.data.lightcurves import load_detrended
from exovet.data.toi import row_to_candidate
from exovet.features import compute_features

log = logging.getLogger(__name__)

DEFAULT_DATASET = Path("data/features.csv")
TARGET_TIMEOUT = 600


class TargetTimeout(BaseException):
    """Deliberately not an Exception: astroquery catches Exception from S3
    downloads and falls back to a second, equally unbounded, MAST download."""


def _raise_timeout(signum, frame):
    raise TargetTimeout


def _features_for(row: pd.Series, author: str, timeout: int) -> dict | None:
    cand = row_to_candidate(row)
    if cand is None:
        return None
    # astroquery downloads have no read timeout, so a stalled connection would
    # block this worker forever. Tasks run on the worker's main thread, so an
    # alarm can interrupt them. It keeps re-firing in case a handler swallows it.
    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout, 30)
    try:
        time, flux, _ = load_detrended(cand, author=author)
        features = compute_features(time, flux, cand)
    except TargetTimeout:
        log.warning("Skipping %s: timed out after %d s", cand.name, timeout)
        return None
    except Exception as exc:  # noqa: BLE001 - network errors, missing data, corrupt files
        log.warning("Skipping %s: %s", cand.name, exc)
        return None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    return {"toi": cand.toi, "tic_id": cand.tic_id, "label": int(row["label"]), **features}


def build_dataset(
    catalog: pd.DataFrame,
    out: Path = DEFAULT_DATASET,
    limit: int | None = None,
    author: str = "SPOC",
    workers: int = 4,
    seed: int = 0,
    timeout: int = TARGET_TIMEOUT,
) -> pd.DataFrame:
    """Compute features for labeled TOIs, appending to ``out`` as it goes.

    TOIs are visited in a seeded random order so a ``limit`` gives a
    representative sample rather than the (brighter, better-observed) earliest
    TOIs. Rows already present in ``out`` are skipped, so an interrupted run
    resumes. Each target gets ``timeout`` seconds before it is skipped.
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
    # Processes rather than threads: lightkurve/astropy FITS reading is not thread-safe.
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_features_for, row, author, timeout) for _, row in labeled.iterrows()
        ]
        for i, future in enumerate(as_completed(futures), 1):
            record = future.result()
            if record is None:
                continue
            pd.DataFrame([record]).to_csv(out, mode="a", header=not out.exists(), index=False)
            log.info("[%d/%d] TOI-%s", i, len(labeled), record["toi"])

    return pd.read_csv(out, dtype={"toi": str})
