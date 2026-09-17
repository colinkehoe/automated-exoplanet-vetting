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
