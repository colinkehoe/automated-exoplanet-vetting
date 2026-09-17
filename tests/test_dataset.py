import os
import socket
import time

import pandas as pd
import pytest
import requests
from requests.adapters import HTTPAdapter

from exovet.data.lightcurves import enforce_http_timeout
from exovet.dataset import _iter_records


# Targets run in spawned processes, so they must be importable module-level functions.
def fake_target(row, author):
    if row["TOI"] == "stall":
        time.sleep(60)
    if row["TOI"] == "crash":
        os._exit(3)
    return {"toi": row["TOI"], "author": author}


def test_iter_records_kills_stalled_and_survives_crashed_targets():
    rows = [pd.Series({"TOI": toi}) for toi in ("stall", "crash", "a", "b", "c")]
    start = time.monotonic()
    records = list(_iter_records(rows, "SPOC", workers=2, timeout=3, target=fake_target))
    assert time.monotonic() - start < 20
    assert records.count(None) == 2
    assert sorted(r["toi"] for r in records if r) == ["a", "b", "c"]
    assert all(r["author"] == "SPOC" for r in records if r)


@pytest.fixture
def recorded_timeouts(monkeypatch):
    seen = []

    def fake_send(self, request, *args, timeout=None, **kwargs):
        seen.append(timeout)
        raise requests.ConnectionError("not sent")

    monkeypatch.setattr(HTTPAdapter, "send", fake_send)
    return seen


def test_default_timeout_applied_when_caller_passes_none(recorded_timeouts):
    enforce_http_timeout((1, 2))
    enforce_http_timeout((1, 2))  # idempotent: must not wrap twice
    for timeout in (None, 5):
        with pytest.raises(requests.ConnectionError):
            requests.get("https://example.invalid", timeout=timeout)
    assert recorded_timeouts == [(1, 2), 5]
    assert not hasattr(HTTPAdapter.send.__wrapped__, "__wrapped__")


def test_stalled_server_times_out(monkeypatch):
    monkeypatch.setattr(HTTPAdapter, "send", HTTPAdapter.send)  # restored afterwards
    with socket.create_server(("127.0.0.1", 0)) as server:  # accepts, never replies
        port = server.getsockname()[1]
        enforce_http_timeout((1, 1))
        start = time.monotonic()
        with pytest.raises(requests.ReadTimeout):
            requests.get(f"http://127.0.0.1:{port}", timeout=None)
        assert time.monotonic() - start < 5


def test_build_dataset_survives_every_target_failing(tmp_path, monkeypatch):
    import pandas as pd

    from exovet import dataset

    monkeypatch.setattr(dataset, "_iter_records", lambda *a, **k: iter([None, None]))
    catalog = pd.DataFrame(
        {"TOI": ["1.01", "2.01"], "TIC ID": [1, 2], "label": [1.0, 0.0], "Detection": ["SPOC"] * 2}
    )
    out = tmp_path / "features.csv"
    assert dataset.build_dataset(catalog, out=out).empty
    assert not out.exists()
