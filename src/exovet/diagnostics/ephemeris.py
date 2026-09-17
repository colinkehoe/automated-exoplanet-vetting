"""Re-fit the ephemeris instead of trusting the catalogue.

Every other diagnostic folds the light curve on the catalogued period and
epoch. If those are wrong the diagnostics measure the wrong thing, and a
common way for them to be wrong is by a factor of two: an eclipsing binary
alerted at half its true period, or a transit whose alternate events were
missed. A box search around the catalogued period, and around half and double
it, says how well the catalogue entry actually fits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from exovet.candidate import Candidate
from exovet.diagnostics.folding import robust_std

PERIOD_WINDOW = 0.02  # fractional half-width of the period search
PERIOD_STEPS = 121
DURATION_FACTORS = (0.5, 0.75, 1.0, 1.5)
MIN_POINTS = 200
MIN_TRANSITS = 2


@dataclass(frozen=True)
class BoxFit:
    """Best box transit found near a trial period."""

    period: float
    epoch: float
    duration: float
    depth: float
    snr: float


def fit_box(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    duration: float,
    window: float = PERIOD_WINDOW,
    steps: int = PERIOD_STEPS,
) -> BoxFit | None:
    """Search a narrow band of periods for the strongest box signal.

    None when the data cannot support the trial period — too few points, or a
    baseline too short to hold ``MIN_TRANSITS`` of it.
    """
    from astropy.timeseries import BoxLeastSquares

    if not (period > 0 and duration > 0) or time.size < MIN_POINTS:
        return None
    if time.max() - time.min() < MIN_TRANSITS * period:
        return None

    sigma = robust_std(flux)
    if not (sigma > 0):
        return None
    periods = np.linspace(period * (1 - window), period * (1 + window), steps)
    durations = np.clip(np.array(DURATION_FACTORS) * duration, 1e-3, 0.4 * period)

    result = BoxLeastSquares(time, flux, dy=sigma).power(periods, durations, objective="snr")
    power = np.asarray(result.power, dtype=float)
    if not np.any(np.isfinite(power)):
        return None
    best = int(np.nanargmax(power))
    return BoxFit(
        period=float(result.period[best]),
        epoch=float(result.transit_time[best]),
        duration=float(result.duration[best]),
        depth=float(result.depth[best]),
        snr=float(result.depth_snr[best]),
    )


def _phase_shift_hours(fit: BoxFit, cand: Candidate) -> float:
    """How far the fitted mid-transit sits from the catalogued one, in hours."""
    offset = (fit.epoch - cand.epoch + 0.5 * fit.period) % fit.period - 0.5 * fit.period
    return offset * 24


def ephemeris_features(time: np.ndarray, flux: np.ndarray, cand: Candidate) -> dict[str, float]:
    best = fit_box(time, flux, cand.period, cand.duration)
    half = fit_box(time, flux, cand.period / 2, cand.duration)
    double = fit_box(time, flux, cand.period * 2, cand.duration)

    def ratio(other: BoxFit | None) -> float:
        if other is None or best is None or best.snr <= 0:
            return math.nan
        return other.snr / best.snr

    if best is None:
        return {
            "refit_snr": math.nan,
            "refit_period_shift": math.nan,
            "refit_epoch_shift_hours": math.nan,
            "refit_duration_ratio": math.nan,
            "refit_depth_ratio": math.nan,
            "alias_half_snr_ratio": math.nan,
            "alias_double_snr_ratio": math.nan,
        }
    return {
        "refit_snr": best.snr,
        "refit_period_shift": (best.period - cand.period) / cand.period,
        "refit_epoch_shift_hours": _phase_shift_hours(best, cand),
        "refit_duration_ratio": best.duration / cand.duration,
        "refit_depth_ratio": best.depth / cand.depth if cand.depth > 0 else math.nan,
        # A signal that is just as strong at half or double the period is not
        # pinned down; an eclipsing binary alerted at half its period is the
        # classic case.
        "alias_half_snr_ratio": ratio(half),
        "alias_double_snr_ratio": ratio(double),
    }
