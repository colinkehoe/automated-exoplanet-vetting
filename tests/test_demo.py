import json

import numpy as np
import pandas as pd
import pytest

from exovet.demo import export
from exovet.model import fit

MAX_BYTES = 2_000_000  # the page has to load over a phone connection


@pytest.fixture
def pieces():
    rng = np.random.default_rng(0)
    n = 60
    label = rng.integers(0, 2, n)
    dataset = pd.DataFrame(
        {
            "toi": [f"{i}.01" for i in range(n)],
            "tic_id": np.arange(n),
            "label": label,
            "star_teff": rng.normal(5000, 500, n),
            "star_radius": rng.normal(1, 0.2, n),
            "depth_snr": np.where(label == 1, rng.normal(30, 5, n), rng.normal(8, 3, n)),
        }
    )
    ranked = dataset.drop(columns="label").head(5).copy()
    ranked["planet_probability"] = np.linspace(0.99, 0.2, 5)
    catalog = pd.DataFrame(
        {
            "TOI": dataset["toi"],
            "TIC ID": dataset["tic_id"],
            "Comments": ["a note"] * n,
            "TESS Disposition": ["CP"] + ["PC"] * (n - 1),
            "Date TOI Alerted (UTC)": ["2019-01-01"] * (n // 2) + ["2022-01-01"] * (n - n // 2),
            "Period (days)": 3.0,
            "Epoch (BJD)": 2459000.0,
            "Duration (hours)": 2.0,
            "Depth (ppm)": 1000.0,
            "label": label.astype(float),
        }
    )
    return dataset, ranked, catalog, fit(dataset)


def test_export_writes_json_the_page_can_read(tmp_path, pieces):
    dataset, ranked, catalog, model = pieces
    paths = export(tmp_path, dataset, ranked, catalog, model, curves=2)
    data = json.loads(paths["data"].read_text())

    assert data["training"]["n"] == len(dataset)
    assert len(data["candidates"]) == len(ranked)
    first = data["candidates"][0]
    assert first["rank"] == 1 and first["toi"] == ranked.iloc[0]["toi"]
    assert first["factors"] and {"feature", "value", "effect"} <= set(first["factors"][0])
    assert data["known"][0]["disposition"] == "CP"
    for key in ("grouped_cv", "temporal", "calibration", "strata"):
        assert key in data["metrics"]
    # No NaN: json.loads accepts it, but JSON.parse in a browser does not.
    assert "NaN" not in paths["data"].read_text()
    assert json.loads(paths["curves"].read_text()) == {}  # no cached light curves in a test


def test_export_stays_small_enough_to_ship(tmp_path, pieces):
    dataset, ranked, catalog, model = pieces
    paths = export(tmp_path, dataset, ranked, catalog, model, curves=2)
    assert sum(p.stat().st_size for p in paths.values()) < MAX_BYTES
