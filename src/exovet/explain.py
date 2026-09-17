"""Why a candidate scored the way it did.

Each feature's effect is how much the evidence for a planet would drop if that
feature took the value of a typical training candidate instead of its observed
one: positive means the observed value argued for a planet, negative against.

Effects are measured in log-odds, not probability. Once a candidate scores 0.003
no single feature can move the probability, yet the evidence behind that score
still divides up meaningfully. A single median makes a poor baseline here,
because features such as the secondary-eclipse depth are bimodal and their
median already looks planet-like.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from exovet.model import VettingModel

EPS = 1e-6


def _log_odds(proba: np.ndarray) -> np.ndarray:
    p = np.clip(proba, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def feature_effects(model: VettingModel, features: pd.Series) -> pd.Series:
    """Effect of each feature on one candidate's probability, largest first."""
    if model.background is None:
        raise ValueError("model was trained before backgrounds were stored; retrain it")
    row = features[model.feature_names]
    background = model.background
    # The candidate, then one copy per (feature, background row) with that
    # feature swapped for the background value.
    swapped = pd.concat([background.assign(**row.drop(name)) for name in model.feature_names])
    evidence = _log_odds(
        model.predict_proba(pd.concat([row.to_frame().T, swapped], ignore_index=True))
    )
    effects = pd.Series(
        evidence[0] - evidence[1:].reshape(len(model.feature_names), len(background)).mean(axis=1),
        index=model.feature_names,
    )
    return effects.reindex(effects.abs().sort_values(ascending=False).index)


def top_factors(model: VettingModel, features: pd.Series, n: int = 3) -> list[dict]:
    """The ``n`` features that moved the evidence most, as plain records."""
    effects = feature_effects(model, features).head(n)
    return [
        {"feature": name, "value": float(features[name]), "effect": float(effect)}
        for name, effect in effects.items()
    ]


def describe(factors: list[dict]) -> str:
    """One-line summary, e.g. ``secondary_snr=8.1 (-3.2); star_radius=2.3 (-1.1)``.

    Effects in brackets are log-odds.
    """
    return "; ".join(f"{f['feature']}={f['value']:.3g} ({f['effect']:+.1f})" for f in factors)
