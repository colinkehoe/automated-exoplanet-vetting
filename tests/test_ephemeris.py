import math

import numpy as np
import pytest
from conftest import synthetic_lightcurve

from exovet.candidate import Candidate
from exovet.diagnostics.ephemeris import ephemeris_features, fit_box


def test_recovers_a_shifted_catalogue_period(cand):
    time, flux = synthetic_lightcurve(cand, days=60.0)
    # Catalogue entry 1% off in period and half an hour off in epoch.
    wrong = Candidate(
        tic_id=cand.tic_id,
        period=cand.period * 1.01,
        epoch=cand.epoch + 0.02,
        duration=cand.duration,
        depth=cand.depth,
    )
    fit = fit_box(time, flux, wrong.period, wrong.duration)
    assert fit.period == pytest.approx(cand.period, rel=2e-3)
    assert fit.snr > 10

    features = ephemeris_features(time, flux, wrong)
    assert features["refit_period_shift"] == pytest.approx(-0.01, abs=2e-3)
    assert abs(features["refit_epoch_shift_hours"]) > 0.1
    assert features["refit_depth_ratio"] == pytest.approx(1.0, rel=0.2)


def test_correct_ephemeris_needs_no_shift(cand):
    time, flux = synthetic_lightcurve(cand, days=60.0)
    features = ephemeris_features(time, flux, cand)
    assert features["refit_period_shift"] == pytest.approx(0.0, abs=1e-3)
    assert features["refit_epoch_shift_hours"] == pytest.approx(0.0, abs=0.2)
    assert features["refit_duration_ratio"] == pytest.approx(1.0, rel=0.6)
    # Half the period overlays transit on flat baseline, so it fits worse.
    assert features["alias_half_snr_ratio"] < 0.9


def test_alternating_depths_favour_the_double_period(cand):
    # An eclipsing binary alerted at half its period: alternate eclipses differ.
    time, flux = synthetic_lightcurve(cand, odd_depth=cand.depth / 3, days=60.0)
    features = ephemeris_features(time, flux, cand)
    assert features["alias_double_snr_ratio"] > 1.0


def test_no_fit_when_the_baseline_is_too_short():
    cand = Candidate(tic_id=1, period=40.0, epoch=5.0, duration=0.2, depth=1e-3)
    time, flux = synthetic_lightcurve(cand, days=25.0)
    features = ephemeris_features(time, flux, cand)
    assert all(math.isnan(v) for v in features.values())
    assert fit_box(np.array([0.0, 1.0]), np.array([1.0, 1.0]), 3.0, 0.1) is None
