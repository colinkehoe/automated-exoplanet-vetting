import math

import numpy as np
import pytest

from exovet.diagnostics.centroid import CentroidSeries, centroid_features
from exovet.diagnostics.folding import in_transit_mask

JITTER = 0.002  # pixels per cadence


def centroid_series(cand, shift=(0.0, 0.0), start=0.0, days=27.0, seed=0):
    rng = np.random.default_rng(seed)
    time = np.arange(start, start + days, 2 / 1440)
    drift = 0.05 * np.sin(time / 3)  # slow pointing drift, cancelled by local baselines
    col = 300 + drift + rng.normal(0, JITTER, time.size)
    row = 500 - drift + rng.normal(0, JITTER, time.size)
    in_transit = in_transit_mask(time, cand)
    col[in_transit] += shift[0]
    row[in_transit] += shift[1]
    return CentroidSeries(time, col, row)


def test_no_shift_is_insignificant(cand):
    sectors = [centroid_series(cand, seed=s, start=30 * s) for s in range(3)]
    f = centroid_features(sectors, cand)
    assert abs(f["centroid_shift_z"]) < 3
    assert f["centroid_shift_pix"] < 5e-4


def test_blend_shift_is_detected_and_located(cand):
    # A source 2 pixels (42") away producing the full 0.2% dip moves the centroid
    # by depth * distance = 0.004 px; sector orientations differ.
    sectors = [
        centroid_series(cand, shift=(0.004, 0.0), seed=0),
        centroid_series(cand, shift=(0.0, -0.004), start=30, seed=1),
    ]
    f = centroid_features(sectors, cand)
    assert f["centroid_shift_z"] > 10
    assert f["centroid_shift_pix"] == pytest.approx(0.004, rel=0.15)
    assert f["centroid_offset_arcsec"] == pytest.approx(42, rel=0.15)


def test_no_usable_transits_gives_nan(cand):
    empty = CentroidSeries(np.array([]), np.array([]), np.array([]))
    f = centroid_features([empty], cand)
    assert all(math.isnan(v) for v in f.values())
