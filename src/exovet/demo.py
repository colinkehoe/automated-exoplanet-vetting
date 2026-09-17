"""Export the ranked candidates, their light curves and the evidence as JSON.

Feeds the standalone demo page, which is a static file with no server behind
it, so everything it shows has to be precomputed here.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from exovet.candidate import Candidate
from exovet.data.lightcurves import detrend, load_cached
from exovet.data.toi import row_to_candidate
from exovet.diagnostics.folding import phase_offset
from exovet.evaluate import evaluate
from exovet.explain import top_factors
from exovet.model import VettingModel

DEFAULT_OUT = Path("portfolio-demo")
CURVE_COUNT = 150
RAW_POINTS = 300  # scatter points kept per curve, to bound the file size
BINS = 80  # upper bound; noisy curves get wider bins so the median is steady
MIN_PER_BIN = 15
ORBIT_BINS = 70  # full-orbit view, where a secondary eclipse shows up
WINDOW_DURATIONS = 6  # half-width of the plotted window, in transit durations

# Shown beside each candidate; the rest of the features stay in the CSV.
HIGHLIGHT_FEATURES = [
    "depth_snr",
    "n_transits",
    "noise_ppm",
    "odd_even_sigma",
    "secondary_max_snr",
    "shape_inner_ratio",
    "centroid_shift_z",
    "log_duration_ratio",
    "star_teff",
    "star_radius",
    "star_distance",
    "tess_mag",
]


def _clean(value):
    """JSON has no NaN or numpy scalars."""
    if isinstance(value, (np.integer, int)):
        return int(value)
    number = float(value)
    return None if not np.isfinite(number) else round(number, 6)


def folded_curve(cand: Candidate) -> dict | None:
    """Phase-folded light curve around transit, as points and a binned median.

    Points are integer pairs of (minutes from mid-transit, parts per million
    below the baseline), which keeps the file small enough to ship with a web
    page; the reader multiplies back out.
    """
    try:
        lc = detrend(load_cached(cand), cand)
    except Exception:  # noqa: BLE001 - target not in the cache, or unusable
        return None
    time = np.asarray(lc.time.value, dtype=float)
    flux = np.asarray(lc.flux.value, dtype=float)
    good = np.isfinite(time) & np.isfinite(flux)
    hours = phase_offset(time[good], cand.period, cand.epoch) * 24
    flux = flux[good]
    flux_all = flux  # before the in-transit window is cut, for the orbit view

    half_width = WINDOW_DURATIONS * cand.duration * 24
    near = np.abs(hours) < half_width
    hours, flux = hours[near], flux[near]
    if hours.size < 240:
        return None

    bins = int(np.clip(hours.size // MIN_PER_BIN, 24, BINS))
    edges = np.linspace(-half_width, half_width, bins + 1)
    index = np.clip(np.digitize(hours, edges) - 1, 0, bins - 1)
    binned = pd.Series(flux).groupby(index).median()
    centres = (edges[:-1] + edges[1:]) / 2

    step = max(1, hours.size // RAW_POINTS)
    order = np.argsort(hours)[::step]

    def encode(phase_hours, values):
        return [
            [round(float(h) * 60), round((float(f) - 1) * 1e6)]
            for h, f in zip(phase_hours, values)
            if np.isfinite(f)
        ]

    # Full orbit, so an eclipse half a period away is visible rather than
    # implied by a number.
    phase = phase_offset(time[good], cand.period, cand.epoch) / cand.period
    orbit_edges = np.linspace(-0.5, 0.5, ORBIT_BINS + 1)
    orbit_index = np.clip(np.digitize(phase, orbit_edges) - 1, 0, ORBIT_BINS - 1)
    orbit = pd.Series(flux_all).groupby(orbit_index).median()
    orbit_centres = (orbit_edges[:-1] + orbit_edges[1:]) / 2

    return {
        "half_width_hours": round(half_width, 4),
        "orbit": [
            [round(float(orbit_centres[i]) * 1000), round((float(v) - 1) * 1e6)]
            for i, v in zip(orbit.index, orbit.to_numpy())
            if np.isfinite(v)
        ],
        "depth_ppm": round(cand.depth * 1e6),
        "points": encode(hours[order], flux[order]),
        "binned": encode(centres[binned.index], binned.to_numpy()),
    }


def candidate_records(
    ranked: pd.DataFrame, model: VettingModel, catalog: pd.DataFrame
) -> list[dict]:
    notes = catalog.set_index("TOI")["Comments"]
    records = []
    for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
        note = notes.get(row["toi"])
        records.append(
            {
                "rank": rank,
                "toi": row["toi"],
                "tic_id": int(row["tic_id"]),
                "probability": round(float(row["planet_probability"]), 4),
                "note": None if pd.isna(note) else str(note)[:160],
                "features": {f: _clean(row[f]) for f in HIGHLIGHT_FEATURES if f in row},
                "factors": [
                    {
                        "feature": f["feature"],
                        "value": _clean(f["value"]),
                        "effect": round(f["effect"], 3),
                    }
                    for f in top_factors(model, row, n=4)
                ],
            }
        )
    return records


def known_dispositions(ranked: pd.DataFrame, catalog: pd.DataFrame) -> list[dict]:
    """Candidates the TESS team already called, which the model never trained on."""
    tess = catalog.set_index("TOI")["TESS Disposition"]
    out = []
    for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
        label = tess.get(row["toi"])
        if isinstance(label, str) and label in {"CP", "KP", "EB", "FP"}:
            out.append(
                {
                    "rank": rank,
                    "toi": row["toi"],
                    "probability": round(float(row["planet_probability"]), 4),
                    "disposition": label,
                }
            )
    return out


def export(
    out: Path,
    dataset: pd.DataFrame,
    ranked: pd.DataFrame,
    catalog: pd.DataFrame,
    model: VettingModel,
    metrics: dict | None = None,
    curves: int = CURVE_COUNT,
) -> dict[str, Path]:
    """Write data.json and curves.json, returning the paths written."""
    out.mkdir(parents=True, exist_ok=True)
    report = metrics if metrics is not None else evaluate(dataset, catalog, seeds=2)
    labels = dataset["label"].astype(int)

    data = {
        "generated": datetime.now(UTC).strftime("%Y-%m-%d"),
        "training": {
            "n": len(dataset),
            "n_planets": int(labels.sum()),
            "n_false_positives": int((1 - labels).sum()),
        },
        "metrics": {
            "grouped_cv": report["grouped_cv"],
            "temporal": report["temporal"],
            "calibration": report["calibration"].reset_index().to_dict("records"),
            "strata": report["strata"].reset_index().to_dict("records"),
        },
        "known": known_dispositions(ranked, catalog),
        "candidates": candidate_records(ranked, model, catalog),
    }

    by_toi = catalog.set_index("TOI")
    folded = {}
    for row in data["candidates"][:curves] + data["candidates"][-10:]:
        entry = by_toi.loc[row["toi"]].copy()
        entry["TOI"] = row["toi"]
        cand = row_to_candidate(entry)
        curve = folded_curve(cand) if cand is not None else None
        if curve:
            curve["period"] = round(cand.period, 5)
            curve["duration_hours"] = round(cand.duration * 24, 3)
            folded[row["toi"]] = curve

    paths = {"data": out / "data.json", "curves": out / "curves.json"}
    paths["data"].write_text(json.dumps(data, separators=(",", ":")))
    paths["curves"].write_text(json.dumps(folded, separators=(",", ":")))
    return paths
