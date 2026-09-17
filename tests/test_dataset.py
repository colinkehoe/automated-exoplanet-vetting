import socket
import time

import pytest
import requests
from requests.adapters import HTTPAdapter

from exovet.data.lightcurves import enforce_http_timeout


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
