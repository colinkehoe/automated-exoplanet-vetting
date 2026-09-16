"""Odd/even transit depth comparison.

An eclipsing binary with equal-ish components detected at half its true period
shows alternating primary and secondary eclipses of different depth.
"""

from __future__ import annotations

import numpy as np

from exovet.candidate import Candidate
from exovet.diagnostics.folding import in_transit_mask, measure_depth, transit_number


def odd_even_features(time: np.ndarray, flux: np.ndarray, cand: Candidate) -> dict[str, float]:
    in_mask = in_transit_mask(time, cand)
    out_mask = ~in_transit_mask(time, cand, width=2.0)
    odd = transit_number(time, cand.period, cand.epoch) % 2 == 1

    d_odd = measure_depth(flux, in_mask & odd, out_mask)
    d_even = measure_depth(flux, in_mask & ~odd, out_mask)
    diff = d_odd.value - d_even.value
    sigma = np.hypot(d_odd.error, d_even.error)
    mean = 0.5 * (d_odd.value + d_even.value)
    return {
        "odd_even_sigma": float(abs(diff) / sigma) if sigma > 0 else float("nan"),
        "odd_even_ratio": float(abs(diff) / mean) if mean > 0 else float("nan"),
    }
