import time

import pandas as pd
import pytest

from exovet import dataset


def _swallow_errors(cand, author):
    # Like astroquery's S3-to-MAST fallback: catch Exception, then stall again.
    try:
        time.sleep(30)
    except Exception:  # noqa: BLE001, S110
        pass
    time.sleep(30)


@pytest.mark.parametrize("stall", [lambda cand, author: time.sleep(30), _swallow_errors])
def test_stalled_target_is_skipped(monkeypatch, stall):
    monkeypatch.setattr(dataset, "load_detrended", stall)
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
