"""Build a labeled feature table from dispositioned TOIs."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from exovet.data.lightcurves import load_detrended
from exovet.data.toi import row_to_candidate
from exovet.features import compute_features

log = logging.getLogger(__name__)

DEFAULT_DATASET = Path("data/features.csv")


def build_dataset(
    catalog: pd.DataFrame,
    out: Path = DEFAULT_DATASET,
    limit: int | None = None,
    author: str = "SPOC",
) -> pd.DataFrame:
    """Compute features for labeled TOIs, appending to ``out`` as it goes.

    Rows already present in ``out`` are skipped, so an interrupted run resumes.
    """
    out = Path(out)
    done: set[str] = set()
    if out.exists():
        done = set(pd.read_csv(out, dtype={"toi": str})["toi"])

    labeled = catalog[catalog["label"].notna() & ~catalog["TOI"].isin(done)]
    if limit is not None:
        labeled = labeled.head(limit)

    out.parent.mkdir(parents=True, exist_ok=True)
    for i, (_, row) in enumerate(labeled.iterrows(), 1):
        cand = row_to_candidate(row)
        if cand is None:
            continue
        try:
            time, flux, _ = load_detrended(cand, author=author)
            features = compute_features(time, flux, cand)
        except Exception as exc:  # noqa: BLE001 - network errors, missing data, corrupt files
            log.warning("Skipping %s: %s", cand.name, exc)
            continue
        record = {"toi": cand.toi, "tic_id": cand.tic_id, "label": int(row["label"]), **features}
        pd.DataFrame([record]).to_csv(out, mode="a", header=not out.exists(), index=False)
        log.info("[%d/%d] %s", i, len(labeled), cand.name)

    return pd.read_csv(out, dtype={"toi": str})
