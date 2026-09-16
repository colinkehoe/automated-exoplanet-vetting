"""Phase folding and depth measurement shared by the diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from exovet.candidate import Candidate


def phase_offset(time: np.ndarray, period: float, epoch: float) -> np.ndarray:
    """Time from nearest transit center, in days, within [-P/2, P/2)."""
    return (time - epoch + 0.5 * period) % period - 0.5 * period


def transit_number(time: np.ndarray, period: float, epoch: float) -> np.ndarray:
    return np.round((time - epoch) / period).astype(int)


def in_transit_mask(time: np.ndarray, cand: Candidate, width: float = 1.0) -> np.ndarray:
    """Cadences within ``width`` transit durations centered on each transit."""
    return np.abs(phase_offset(time, cand.period, cand.epoch)) < 0.5 * width * cand.duration


def robust_std(x: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


@dataclass(frozen=True)
class Depth:
    value: float
    error: float
    n_points: int

    @property
    def snr(self) -> float:
        if not np.isfinite(self.error) or self.error <= 0:
            return float("nan")
        return self.value / self.error


def measure_depth(flux: np.ndarray, in_mask: np.ndarray, out_mask: np.ndarray) -> Depth:
    """Box depth: out-of-transit median minus in-transit mean."""
    n_in = int(in_mask.sum())
    if n_in == 0 or out_mask.sum() < 2:
        return Depth(float("nan"), float("nan"), n_in)
    baseline = np.median(flux[out_mask])
    noise = robust_std(flux[out_mask])
    return Depth(float(baseline - flux[in_mask].mean()), noise / np.sqrt(n_in), n_in)
