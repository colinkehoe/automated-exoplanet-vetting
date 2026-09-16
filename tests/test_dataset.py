import time

import pandas as pd

from exovet import dataset


def test_stalled_target_is_skipped(monkeypatch):
    monkeypatch.setattr(dataset, "load_detrended", lambda cand, author: time.sleep(30))
    row = pd.Series(
        {
            "TIC ID": 1,
            "TOI": "1.01",
            "Period (days)": 2.0,
            "Epoch (BJD)": 2459000.0,
            "Duration (hours)": 2.0,
            "Depth (ppm)": 1000.0,
            "label": 1.0,
        }
    )
    start = time.monotonic()
    assert dataset._features_for(row, "SPOC", timeout=1) is None
    assert time.monotonic() - start < 5
