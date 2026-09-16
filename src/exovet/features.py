"""Combine all diagnostics into a single feature vector."""

from __future__ import annotations

import numpy as np

from exovet.candidate import Candidate
from exovet.diagnostics.odd_even import odd_even_features
from exovet.diagnostics.secondary import secondary_features
from exovet.diagnostics.shape import consistency_features, shape_features


def candidate_features(cand: Candidate) -> dict[str, float]:
    return {
        "log_period": float(np.log10(cand.period)),
        "log_depth": float(np.log10(cand.depth)) if cand.depth > 0 else float("nan"),
        "duration_hours": cand.duration * 24.0,
        "duty_cycle": cand.duration / cand.period,
    }


def compute_features(time: np.ndarray, flux: np.ndarray, cand: Candidate) -> dict[str, float]:
    """Compute all features for a candidate from a detrended light curve."""
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    features = candidate_features(cand)
    features.update(shape_features(time, flux, cand))
    features.update(consistency_features(time, flux, cand))
    features.update(odd_even_features(time, flux, cand))
    features.update(secondary_features(time, flux, cand))
    return {k: float(v) for k, v in features.items()}
