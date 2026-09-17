import math

import pandas as pd
import pytest

from exovet.candidate import Candidate, Star
from exovet.dataset import refresh_catalog_features
from exovet.diagnostics.stellar import expected_duration, stellar_features, stellar_mass

SUN = Star(teff=5772.0, logg=4.438, radius=1.0, mass=1.0, distance=10.0, tess_mag=5.0)


def test_earth_sun_duration():
    # Earth crossing the Sun centrally takes about 13 hours.
    assert expected_duration(365.25, 8.4e-5, 1.0, 1.0) * 24 == pytest.approx(13.0, abs=0.3)


def test_mass_from_logg_when_catalog_mass_missing():
    star = Star(logg=4.438, radius=1.0)
    assert stellar_mass(star) == pytest.approx(1.0, rel=0.01)


def test_hot_jupiter_consistent_with_sun_like_star():
    depth = 0.01  # Rp/R* = 0.1, about 1.1 Jupiter radii
    t = expected_duration(3.0, depth, 1.0, 1.0)
    cand = Candidate(tic_id=1, period=3.0, epoch=0.0, duration=t, depth=depth, star=SUN)
    f = stellar_features(cand)
    assert f["log_duration_ratio"] == pytest.approx(0.0, abs=1e-9)
    assert 10 ** f["log_planet_radius"] == pytest.approx(10.9, abs=0.1)
    assert f["star_log_density"] == pytest.approx(0.0)


def test_too_long_duration_flags_inconsistency():
    # A 12 h eclipse at P = 3 d is far too long for a Sun-like host.
    cand = Candidate(tic_id=1, period=3.0, epoch=0.0, duration=0.5, depth=0.01, star=SUN)
    assert stellar_features(cand)["log_duration_ratio"] > 0.5


def test_missing_star_gives_nan():
    cand = Candidate(tic_id=1, period=3.0, epoch=0.0, duration=0.1, depth=0.01)
    f = stellar_features(cand)
    assert set(f) >= {"log_duration_ratio", "log_planet_radius", "star_log_density"}
    assert all(math.isnan(v) for v in f.values())


def _catalog_row(toi, label, radius):
    return {
        "TIC ID": 7,
        "TOI": toi,
        "Period (days)": 3.0,
        "Epoch (BJD)": 2459000.0,
        "Duration (hours)": 3.0,
        "Depth (ppm)": 10000.0,
        "Stellar Radius (R_Sun)": radius,
        "Stellar Mass (M_Sun)": 1.0,
        "label": label,
    }


def test_refresh_adds_catalog_columns_and_updates_labels():
    dataset = pd.DataFrame(
        {
            "toi": ["1.01", "2.01", "9.01"],
            "tic_id": [7, 7, 7],
            "label": [1, 1, 0],
            "log_period": [0.0, 0.0, 0.0],
            "depth_snr": [10.0, 20.0, 30.0],
            "star_radius": [0.5, 0.5, 0.5],
        }
    )
    catalog = pd.DataFrame(
        [_catalog_row("1.01", 1.0, 1.0), _catalog_row("2.01", 0.0, float("nan"))]
    )
    out = refresh_catalog_features(dataset, catalog).set_index("toi")

    assert out.loc["2.01", "label"] == 0  # disposition changed in the catalog
    assert out.loc["9.01", "label"] == 0  # not in catalog: kept as is
    assert out.loc["1.01", "log_period"] == pytest.approx(math.log10(3.0))
    assert out.loc["1.01", "log_planet_radius"] == pytest.approx(math.log10(10.908))
    assert out.loc["1.01", "star_radius"] == 1.0
    assert math.isnan(out.loc["2.01", "star_radius"])  # catalog value removed
    assert out.loc["9.01", "star_radius"] == 0.5
    assert out["depth_snr"].tolist() == [10.0, 20.0, 30.0]
    assert list(out.columns[:3]) == ["tic_id", "label", "log_period"]
    assert list(out.columns[-1:]) == ["depth_snr"]
