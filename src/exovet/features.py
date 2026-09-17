"""Combine all diagnostics into a single feature vector."""

from __future__ import annotations

import numpy as np

from exovet.candidate import Candidate
from exovet.diagnostics.centroid import CentroidSeries, centroid_features
from exovet.diagnostics.ephemeris import ephemeris_features
from exovet.diagnostics.odd_even import odd_even_features
from exovet.diagnostics.secondary import secondary_features
from exovet.diagnostics.shape import consistency_features, shape_features
from exovet.diagnostics.stellar import stellar_features


def candidate_features(cand: Candidate) -> dict[str, float]:
    return {
        "log_period": float(np.log10(cand.period)),
        "log_depth": float(np.log10(cand.depth)) if cand.depth > 0 else float("nan"),
        "duration_hours": cand.duration * 24.0,
        "duty_cycle": cand.duration / cand.period,
    }


def catalog_features(cand: Candidate) -> dict[str, float]:
    """Features that need only the catalog entry, not the light curve."""
    features = candidate_features(cand)
    features.update(stellar_features(cand))
    return {k: float(v) for k, v in features.items()}


def compute_features(
    time: np.ndarray,
    flux: np.ndarray,
    cand: Candidate,
    centroids: list[CentroidSeries] | None = None,
) -> dict[str, float]:
    """Compute all features for a candidate from a detrended light curve.

    ``centroids`` holds each sector's centroid series; without it the centroid
    features are NaN.
    """
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    features = catalog_features(cand)
    features.update(shape_features(time, flux, cand))
    features.update(consistency_features(time, flux, cand))
    features.update(odd_even_features(time, flux, cand))
    features.update(secondary_features(time, flux, cand))
    features.update(centroid_features(centroids or [], cand))
    features.update(ephemeris_features(time, flux, cand))
    return {k: float(v) for k, v in features.items()}
