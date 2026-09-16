"""Transit shape and per-transit consistency.

Grazing eclipsing binaries produce V-shaped dips; planets transiting a star
produce flat-bottomed U shapes. Systematics often masquerade as a periodic
signal but are dominated by one or two events.
"""

from __future__ import annotations

import numpy as np

from exovet.candidate import Candidate
from exovet.diagnostics.folding import (
    in_transit_mask,
    measure_depth,
    phase_offset,
    robust_std,
    transit_number,
)


def shape_features(time: np.ndarray, flux: np.ndarray, cand: Candidate) -> dict[str, float]:
    offset = np.abs(phase_offset(time, cand.period, cand.epoch))
    out_mask = offset > cand.duration
    full = measure_depth(flux, offset < 0.5 * cand.duration, out_mask)
    inner = measure_depth(flux, offset < 0.25 * cand.duration, out_mask)
    # Ingress/egress: the outer half of the transit window.
    outer = measure_depth(
        flux, (offset >= 0.25 * cand.duration) & (offset < 0.5 * cand.duration), out_mask
    )

    def div(a, b):
        return a / b if b > 0 else float("nan")

    return {
        "depth_snr": full.snr,
        "depth_ratio_catalog": div(full.value, cand.depth),
        "shape_inner_ratio": div(inner.value, full.value),
        "shape_outer_ratio": div(outer.value, inner.value),
        "noise_ppm": robust_std(flux[out_mask]) * 1e6,
    }


def consistency_features(time: np.ndarray, flux: np.ndarray, cand: Candidate) -> dict[str, float]:
    in_mask = in_transit_mask(time, cand)
    out_mask = ~in_transit_mask(time, cand, width=2.0)
    numbers = transit_number(time, cand.period, cand.epoch)

    depths, errors = [], []
    for n in np.unique(numbers[in_mask]):
        d = measure_depth(flux, in_mask & (numbers == n), out_mask)
        if np.isfinite(d.value) and np.isfinite(d.error) and d.error > 0:
            depths.append(d.value)
            errors.append(d.error)
    depths, errors = np.array(depths), np.array(errors)

    n = len(depths)
    result = {
        "n_transits": float(n),
        "depth_chi2_reduced": float("nan"),
        "max_single_transit_fraction": float("nan"),
    }
    if n >= 2:
        weights = errors**-2
        mean = np.sum(weights * depths) / np.sum(weights)
        result["depth_chi2_reduced"] = float(np.sum(((depths - mean) / errors) ** 2) / (n - 1))
    positive = np.clip(depths, 0, None)
    if n >= 1 and positive.sum() > 0:
        result["max_single_transit_fraction"] = float(positive.max() / positive.sum())
    return result
