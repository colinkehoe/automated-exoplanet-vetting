import numpy as np
import pandas as pd
import pytest

from exovet.explain import describe, feature_effects, top_factors
from exovet.model import fit


@pytest.fixture
def model():
    rng = np.random.default_rng(0)
    n = 300
    label = rng.integers(0, 2, n)
    dataset = pd.DataFrame(
        {
            "toi": [f"{i}.01" for i in range(n)],
            "tic_id": np.arange(n),
            "label": label,
            # Only this feature carries signal; large values mean false positive.
            "secondary_snr": np.where(label == 1, rng.normal(0, 1, n), rng.normal(10, 2, n)),
            "noise_ppm": rng.normal(100, 10, n),
        }
    )
    return fit(dataset)


def test_effects_point_at_the_informative_feature(model):
    fp_like = pd.Series({"secondary_snr": 12.0, "noise_ppm": 100.0})
    effects = feature_effects(model, fp_like)
    assert effects.index[0] == "secondary_snr"
    assert effects["secondary_snr"] < -1  # argued against a planet, in log-odds
    assert abs(effects["noise_ppm"]) < 1


def test_effect_sign_follows_the_evidence(model):
    planet_like = pd.Series({"secondary_snr": 0.0, "noise_ppm": 100.0})
    assert feature_effects(model, planet_like)["secondary_snr"] > 1


def test_top_factors_and_description(model):
    factors = top_factors(model, pd.Series({"secondary_snr": 12.0, "noise_ppm": 100.0}), n=1)
    assert factors[0]["feature"] == "secondary_snr"
    assert factors[0]["value"] == 12.0
    assert describe(factors).startswith("secondary_snr=12 (-")
