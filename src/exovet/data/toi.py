"""TESS Objects of Interest catalog from ExoFOP."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from exovet.candidate import BTJD_OFFSET, Candidate

TOI_URL = "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"
DEFAULT_CACHE = Path("cache/toi.csv")

# TFOPWG dispositions with a definite answer. PC/APC/unset are unlabeled.
LABELS = {"CP": 1, "KP": 1, "FP": 0, "FA": 0}


def load_toi_catalog(path: Path = DEFAULT_CACHE, refresh: bool = False) -> pd.DataFrame:
    """Load the TOI table, downloading it to ``path`` if missing or ``refresh`` is set."""
    path = Path(path)
    if refresh or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.read_csv(TOI_URL).to_csv(path, index=False)
    df = pd.read_csv(path, dtype={"TOI": str})
    df["label"] = df["TFOPWG Disposition"].map(LABELS)
    return df


def row_to_candidate(row: pd.Series) -> Candidate | None:
    """Convert a TOI table row, or return None if its ephemeris is incomplete."""
    values = [row["Period (days)"], row["Epoch (BJD)"], row["Duration (hours)"], row["Depth (ppm)"]]
    if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in values):
        return None
    period, epoch, duration_hr, depth_ppm = (float(v) for v in values)
    if period <= 0 or duration_hr <= 0:
        return None
    return Candidate(
        tic_id=int(row["TIC ID"]),
        period=period,
        epoch=epoch - BTJD_OFFSET,
        duration=duration_hr / 24.0,
        depth=depth_ppm * 1e-6,
        toi=str(row["TOI"]),
    )


def find_toi(df: pd.DataFrame, toi: str) -> Candidate:
    match = df[df["TOI"] == str(toi)]
    if match.empty:
        raise KeyError(f"TOI {toi} not found in catalog")
    cand = row_to_candidate(match.iloc[0])
    if cand is None:
        raise ValueError(f"TOI {toi} has an incomplete ephemeris")
    return cand
