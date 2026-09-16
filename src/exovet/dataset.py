"""Build a labeled feature table from dispositioned TOIs."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from exovet.data.lightcurves import load_detrended
from exovet.data.toi import row_to_candidate
from exovet.features import compute_features

log = logging.getLogger(__name__)

DEFAULT_DATASET = Path("data/features.csv")


def _features_for(row: pd.Series, author: str) -> dict | None:
    cand = row_to_candidate(row)
    if cand is None:
        return None
    try:
        time, flux, _ = load_detrended(cand, author=author)
        features = compute_features(time, flux, cand)
    except Exception as exc:  # noqa: BLE001 - network errors, missing data, corrupt files
        log.warning("Skipping %s: %s", cand.name, exc)
        return None
    return {"toi": cand.toi, "tic_id": cand.tic_id, "label": int(row["label"]), **features}


def build_dataset(
    catalog: pd.DataFrame,
    out: Path = DEFAULT_DATASET,
    limit: int | None = None,
    author: str = "SPOC",
    workers: int = 4,
    seed: int = 0,
) -> pd.DataFrame:
    """Compute features for labeled TOIs, appending to ``out`` as it goes.

    TOIs are visited in a seeded random order so a ``limit`` gives a
    representative sample rather than the (brighter, better-observed) earliest
    TOIs. Rows already present in ``out`` are skipped, so an interrupted run
    resumes.
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
    # Downloads dominate the runtime, so threads parallelize well.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_features_for, row, author) for _, row in labeled.iterrows()]
        for i, future in enumerate(as_completed(futures), 1):
            record = future.result()
            if record is None:
                continue
            pd.DataFrame([record]).to_csv(out, mode="a", header=not out.exists(), index=False)
            log.info("[%d/%d] TOI-%s", i, len(labeled), record["toi"])

    return pd.read_csv(out, dtype={"toi": str})
