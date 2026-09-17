import numpy as np
import pandas as pd
import pytest

from exovet.evaluate import calibration_table, temporal_holdout


def test_calibration_table_bins_and_fractions():
    y = np.array([0, 0, 1, 1, 1])
    proba = np.array([0.05, 0.15, 0.15, 0.95, 1.0])  # 1.0 falls in the top bin
    table = calibration_table(y, proba)
    assert list(table.index) == ["0.0-0.1", "0.1-0.2", "0.9-1.0"]
    assert table.loc["0.1-0.2", "n"] == 2
    assert table.loc["0.1-0.2", "observed"] == pytest.approx(0.5)
    assert table.loc["0.9-1.0", "predicted"] == pytest.approx(0.975)


def test_temporal_holdout_splits_on_alert_date():
    rng = np.random.default_rng(0)
    n = 200
    label = rng.integers(0, 2, n)
    dataset = pd.DataFrame(
        {
            "toi": [f"{i}.01" for i in range(n)],
            "tic_id": np.arange(n),
            "label": label,
            "odd_even_sigma": np.where(label == 1, rng.normal(0, 1, n), rng.normal(6, 2, n)),
        }
    )
    alerted = pd.Series(pd.date_range("2019-01-01", periods=n, freq="D"))
    result = temporal_holdout(dataset, alerted, "2019-05-01")
    assert (result["n_train"], result["n_test"]) == (120, 80)
    assert result["roc_auc"] > 0.9


def test_rank_candidates_orders_and_keeps_columns():
    from exovet.model import fit, rank_candidates

    rng = np.random.default_rng(0)
    n = 120
    label = rng.integers(0, 2, n)
    train = pd.DataFrame(
        {
            "toi": [f"{i}.01" for i in range(n)],
            "tic_id": np.arange(n),
            "label": label,
            "odd_even_sigma": np.where(label == 1, rng.normal(0, 1, n), rng.normal(8, 2, n)),
        }
    )
    model = fit(train)
    candidates = pd.DataFrame(
        {"toi": ["9.01", "8.01"], "tic_id": [1, 2], "odd_even_sigma": [8.0, 0.0]}
    )
    ranked = rank_candidates(model, candidates)
    assert list(ranked.columns[:3]) == ["toi", "tic_id", "planet_probability"]
    assert ranked.toi.tolist() == ["8.01", "9.01"]  # planet-like first
    assert ranked.planet_probability.is_monotonic_decreasing


def test_stratified_scores_reports_each_populated_bin():
    from exovet.evaluate import stratified_scores

    rng = np.random.default_rng(0)
    n = 200
    label = rng.integers(0, 2, n)
    dataset = pd.DataFrame(
        {
            "label": label,
            "star_teff": np.where(np.arange(n) % 2 == 0, 3500.0, 5500.0),
            "star_distance": np.full(n, 50.0),
            "star_radius": np.full(n, 1.0),
        }
    )
    proba = np.where(label == 1, rng.uniform(0.5, 1, n), rng.uniform(0, 0.5, n))
    table = stratified_scores(dataset, proba)
    assert len(table) == 4  # two Teff bins, one distance bin, one radius bin
    assert (table["roc_auc"] > 0.9).all()
    assert table.loc["star_distance (0.0, 100.0]", "n"] == n
