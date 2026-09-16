import math

import numpy as np
import pandas as pd
import pytest
from conftest import synthetic_lightcurve

from exovet.candidate import Candidate
from exovet.data.toi import row_to_candidate
from exovet.diagnostics.folding import phase_offset
from exovet.diagnostics.odd_even import odd_even_features
from exovet.diagnostics.secondary import secondary_features
from exovet.diagnostics.shape import consistency_features, shape_features
from exovet.features import compute_features


def test_phase_offset_range():
    t = np.linspace(-10, 10, 1001)
    off = phase_offset(t, 3.0, 1.0)
    assert off.min() >= -1.5 and off.max() < 1.5
    assert phase_offset(np.array([7.0]), 3.0, 1.0)[0] == pytest.approx(0.0)


def test_planet_like_signal(cand):
    time, flux = synthetic_lightcurve(cand)
    f = compute_features(time, flux, cand)
    assert f["depth_ratio_catalog"] == pytest.approx(1.0, abs=0.05)
    assert f["depth_snr"] > 50
    assert f["odd_even_sigma"] < 3
    assert abs(f["secondary_snr"]) < 3
    assert f["shape_inner_ratio"] == pytest.approx(1.0, abs=0.1)
    assert f["n_transits"] == 18
    assert f["depth_chi2_reduced"] < 3
    assert all(isinstance(v, float) for v in f.values())


def test_odd_even_mismatch(cand):
    time, flux = synthetic_lightcurve(cand, odd_depth=1e-3)
    f = odd_even_features(time, flux, cand)
    assert f["odd_even_sigma"] > 10
    assert f["odd_even_ratio"] == pytest.approx(2 / 3, abs=0.05)


def test_secondary_eclipse(cand):
    time, flux = synthetic_lightcurve(cand, secondary_depth=5e-4)
    f = secondary_features(time, flux, cand)
    assert f["secondary_snr"] > 10
    assert f["secondary_ratio"] == pytest.approx(0.25, abs=0.03)
    assert f["secondary_max_snr"] >= f["secondary_snr"]


def test_v_shape(cand):
    time, flux = synthetic_lightcurve(cand, v_shaped=True)
    f = shape_features(time, flux, cand)
    # Triangle: inner half mean depth is 1.5x the full-window mean.
    assert f["shape_inner_ratio"] == pytest.approx(1.5, abs=0.1)


def test_single_event_dominates(cand):
    time, flux = synthetic_lightcurve(cand)
    flux[np.abs(time - 4.0) < 0.06] -= 0.05  # one huge dip at transit 1
    f = consistency_features(time, flux, cand)
    assert f["max_single_transit_fraction"] > 0.5
    assert f["depth_chi2_reduced"] > 100


def test_no_transits_observed():
    cand = Candidate(tic_id=1, period=100.0, epoch=80.0, duration=0.2, depth=1e-3)
    time, flux = synthetic_lightcurve(cand, days=20.0)
    f = compute_features(time, flux, cand)
    assert f["n_transits"] == 0
    assert math.isnan(f["depth_snr"])
    assert math.isnan(f["odd_even_sigma"])


def test_row_to_candidate():
    row = pd.Series(
        {
            "TIC ID": 12345,
            "TOI": "100.01",
            "Period (days)": 2.5,
            "Epoch (BJD)": 2459000.5,
            "Duration (hours)": 3.0,
            "Depth (ppm)": 1500.0,
        }
    )
    cand = row_to_candidate(row)
    assert cand.epoch == pytest.approx(2000.5)
    assert cand.duration == pytest.approx(0.125)
    assert cand.depth == pytest.approx(1.5e-3)
    assert cand.name == "TOI-100.01"
    assert row_to_candidate(row.replace(2.5, float("nan"))) is None
