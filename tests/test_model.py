import numpy as np
import pandas as pd

from exovet.model import VettingModel, train


def test_train_save_load(tmp_path):
    rng = np.random.default_rng(0)
    n = 200
    label = rng.integers(0, 2, n)
    df = pd.DataFrame(
        {
            "toi": [f"{i}.01" for i in range(n)],
            "tic_id": np.arange(n),
            "label": label,
            "odd_even_sigma": np.where(label == 1, rng.normal(0, 1, n), rng.normal(8, 2, n)),
            "noise_ppm": rng.normal(100, 10, n),
        }
    )
    df.loc[::7, "odd_even_sigma"] = np.nan  # NaN features must be tolerated

    model, metrics = train(df, folds=3)
    assert model.feature_names == ["odd_even_sigma", "noise_ppm"]
    assert metrics["roc_auc"] > 0.8

    path = tmp_path / "m.joblib"
    model.save(path)
    loaded = VettingModel.load(path)
    p = loaded.predict_proba({"noise_ppm": 100.0, "odd_even_sigma": 0.1})
    assert 0 <= p[0] <= 1
