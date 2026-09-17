"""Centroid shift during transit.

If the dip happens on a nearby star, the flux-weighted centroid moves toward
the target while that star dims. The shift divided by the depth estimates how
far away the true source is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from exovet.candidate import Candidate
from exovet.diagnostics.folding import phase_offset, robust_std, transit_number

TESS_PIXEL_ARCSEC = 21.0
MIN_IN, MIN_OUT = 3, 6


@dataclass(frozen=True)
class CentroidSeries:
    """Flux-weighted centroid (column, row) in pixels for one sector."""

    time: np.ndarray
    col: np.ndarray
    row: np.ndarray


def sector_shift(series: CentroidSeries, cand: Candidate) -> tuple[np.ndarray, np.ndarray] | None:
    """In-minus-out centroid shift and its error, per axis, for one sector.

    Each transit is compared with its own nearby baseline, so slow pointing
    drift cancels.
    """
    xy = np.vstack([series.col, series.row])
    good = np.isfinite(series.time) & np.all(np.isfinite(xy), axis=0)
    time, xy = series.time[good], xy[:, good]

    offset = np.abs(phase_offset(time, cand.period, cand.epoch))
    numbers = transit_number(time, cand.period, cand.epoch)
    in_mask = offset < 0.5 * cand.duration
    out_mask = (offset > cand.duration) & (offset < 3 * cand.duration)

    total = np.zeros(2)
    n_in = 0
    residuals = []
    for n in np.unique(numbers[in_mask]):
        i = in_mask & (numbers == n)
        o = out_mask & (numbers == n)
        if i.sum() < MIN_IN or o.sum() < MIN_OUT:
            continue
        baseline = np.median(xy[:, o], axis=1, keepdims=True)
        total += (xy[:, i] - baseline).sum(axis=1)
        n_in += int(i.sum())
        residuals.append(xy[:, o] - baseline)
    if n_in == 0:
        return None

    residuals = np.hstack(residuals)
    sigma = np.array([robust_std(residuals[0]), robust_std(residuals[1])]) / np.sqrt(n_in)
    if not np.all(sigma > 0):
        return None
    return total / n_in, sigma


def centroid_features(sectors: list[CentroidSeries], cand: Candidate) -> dict[str, float]:
    shifts = [s for s in (sector_shift(series, cand) for series in sectors) if s is not None]
    if not shifts:
        return {
            "centroid_shift_z": math.nan,
            "centroid_shift_pix": math.nan,
            "centroid_offset_arcsec": math.nan,
        }

    # Pixel axes are oriented differently in each sector, so combine
    # significance as a chi-square and size as a weighted mean magnitude.
    chi2 = sum(float(np.sum((d / s) ** 2)) for d, s in shifts)
    dof = 2 * len(shifts)
    weights = np.array([1 / np.sum(s**2) for _, s in shifts])
    sizes = np.array([np.hypot(*d) for d, _ in shifts])
    shift = float(np.sum(weights * sizes) / np.sum(weights))
    return {
        "centroid_shift_z": (chi2 - dof) / math.sqrt(2 * dof),
        "centroid_shift_pix": shift,
        "centroid_offset_arcsec": (
            shift * TESS_PIXEL_ARCSEC / cand.depth if cand.depth > 0 else math.nan
        ),
    }
