import numpy as np
import pytest

from exovet.data import lightcurves


class FakeSearch(list):
    """Stands in for lightkurve's SearchResult: a sized, sliceable result."""

    def __getitem__(self, item):
        return (
            FakeSearch(super().__getitem__(item))
            if isinstance(item, slice)
            else list.__getitem__(self, item)
        )

    def download_all(self, **kwargs):
        return self


class FakeLightCurve:
    def __init__(self, centroids=True, nan_centroids=False):
        self.quality = np.zeros(5, dtype=int)
        self.time = type("T", (), {"value": np.arange(5.0)})()
        if centroids:
            values = np.full(5, np.nan) if nan_centroids else np.arange(5.0)
            self.centroid_col = type("C", (), {"value": values})()
            self.centroid_row = type("C", (), {"value": values + 10})()


def test_falls_back_to_next_pipeline(monkeypatch):
    searched = []

    def fake_search(tic_id, author):
        searched.append(author)
        return FakeSearch([FakeLightCurve()] if author == "QLP" else [])

    monkeypatch.setattr(lightcurves, "search_sectors", fake_search)
    sectors = lightcurves.fetch_sectors(1, authors=("SPOC", "QLP"), max_sectors=None)
    assert searched == ["SPOC", "QLP"]
    assert len(sectors) == 1


def test_raises_when_no_pipeline_has_data(monkeypatch):
    monkeypatch.setattr(lightcurves, "search_sectors", lambda tic_id, author: FakeSearch())
    with pytest.raises(LookupError, match="SPOC/QLP"):
        lightcurves.fetch_sectors(1, authors=("SPOC", "QLP"))


def test_sector_cap_limits_downloads(monkeypatch):
    monkeypatch.setattr(
        lightcurves, "search_sectors", lambda tic_id, author: FakeSearch([FakeLightCurve()] * 25)
    )
    assert len(lightcurves.fetch_sectors(1, max_sectors=10)) == 10


@pytest.mark.parametrize(
    ("lc", "expected"),
    [
        (FakeLightCurve(), True),
        (FakeLightCurve(centroids=False), False),  # QLP: no centroid columns
        (FakeLightCurve(nan_centroids=True), False),
    ],
)
def test_centroid_series_present_only_when_usable(lc, expected):
    assert (lightcurves.centroid_series(lc) is not None) == expected
