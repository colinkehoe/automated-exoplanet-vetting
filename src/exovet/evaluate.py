"""Evaluation beyond a single cross-validation score."""

from __future__ import annotations

import numpy as np
import pandas as pd

from exovet.model import cross_validated_proba, fit, score

ALERT_COLUMN = "Date TOI Alerted (UTC)"


def alert_dates(dataset: pd.DataFrame, catalog: pd.DataFrame) -> pd.Series:
    dates = catalog.set_index("TOI")[ALERT_COLUMN]
    return pd.to_datetime(dataset["toi"].map(dates))


def calibration_table(y: np.ndarray, proba: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Predicted probability vs. observed planet fraction, per probability bin."""
    edges = np.linspace(0, 1, bins + 1)
    index = np.clip(np.digitize(proba, edges) - 1, 0, bins - 1)
    table = pd.DataFrame({"bin": index, "predicted": proba, "observed": y}).groupby("bin")
    out = table.agg(
        n=("observed", "size"), predicted=("predicted", "mean"), observed=("observed", "mean")
    )
    out.index = [f"{edges[i]:.1f}-{edges[i + 1]:.1f}" for i in out.index]
    return out


def temporal_holdout(dataset: pd.DataFrame, alerted: pd.Series, cutoff: str) -> dict:
    """Train on TOIs alerted before ``cutoff`` and test on the rest.

    Mirrors real use, where a model trained on past dispositions scores newer,
    typically fainter and noisier, candidates.
    """
    before = (alerted < pd.Timestamp(cutoff)).to_numpy()
    train, test = dataset[before], dataset[~before]
    proba = fit(train).predict_proba(test)
    y = test["label"].astype(int).to_numpy()
    return {
        "cutoff": cutoff,
        "n_train": len(train),
        "n_test": len(test),
        "test_planet_fraction": float(y.mean()),
        "test_mean_probability": float(proba.mean()),
        **score(y, proba),
    }


def evaluate(
    dataset: pd.DataFrame,
    catalog: pd.DataFrame,
    cutoff: str = "2021-01-01",
    folds: int = 5,
    seeds: int = 3,
) -> dict:
    y = dataset["label"].astype(int).to_numpy()
    probas = [cross_validated_proba(dataset, folds, seed) for seed in range(seeds)]
    runs = pd.DataFrame([score(y, p) for p in probas])
    return {
        "grouped_cv": {
            "folds": folds,
            "seeds": seeds,
            "mean": runs.mean().round(4).to_dict(),
            "std": runs.std(ddof=0).round(4).to_dict(),
        },
        "calibration": calibration_table(y, np.mean(probas, axis=0)).round(3),
        "temporal": temporal_holdout(dataset, alert_dates(dataset, catalog), cutoff),
    }
