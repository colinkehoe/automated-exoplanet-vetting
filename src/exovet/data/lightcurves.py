"""Download and detrend TESS light curves via lightkurve."""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np

from exovet.candidate import Candidate
from exovet.diagnostics.centroid import CentroidSeries
from exovet.diagnostics.folding import in_transit_mask

DEFAULT_DOWNLOAD_DIR = Path("cache/lightcurves")
MAX_SECTORS = 10
# (connect, read) seconds; the read timeout bounds a stall, not a whole download.
HTTP_TIMEOUT = (30, 120)


def enforce_http_timeout(timeout: tuple[float, float] = HTTP_TIMEOUT) -> None:
    """Give requests a default timeout wherever the caller passes none.

    astroquery downloads with ``timeout=None``, which disables socket timeouts,
    so a stalled MAST connection blocks forever. Signals cannot reliably break
    that read on macOS, where they may be delivered to another thread.
    """
    from requests.adapters import HTTPAdapter

    original = getattr(HTTPAdapter.send, "__wrapped__", HTTPAdapter.send)

    @functools.wraps(original)
    def send(self, request, *args, timeout=None, **kwargs):
        return original(self, request, *args, timeout=timeout or default, **kwargs)

    default = timeout
    HTTPAdapter.send = send


def fetch_sectors(
    tic_id: int,
    author: str = "SPOC",
    download_dir: Path = DEFAULT_DOWNLOAD_DIR,
    max_sectors: int | None = MAX_SECTORS,
):
    """Download up to ``max_sectors`` sectors for a target, one light curve each.

    SPOC is restricted to 2-minute cadence so sectors stitch cleanly; other
    pipelines (e.g. QLP) take whatever cadence is available. The sector cap
    keeps targets in the continuous viewing zone (~40 sectors) tractable.
    """
    import lightkurve as lk

    enforce_http_timeout()
    exptime = 120 if author == "SPOC" else None
    search = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS", author=author, exptime=exptime)
    if len(search) == 0:
        raise LookupError(f"No {author} light curves for TIC {tic_id}")
    if max_sectors is not None:
        search = search[:max_sectors]
    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    try:
        collection = search.download_all(download_dir=str(download_dir))
    except lk.LightkurveError:
        # A cached file is corrupt (e.g. an interrupted download). lightkurve
        # reads any file already at the cache path, so delete them and refetch.
        for path in _cached_paths(search, download_dir):
            path.unlink(missing_ok=True)
        collection = search.download_all(download_dir=str(download_dir))
    return collection


def centroid_series(lc) -> CentroidSeries:
    """Flux-weighted centroids of one sector, keeping only unflagged cadences."""
    good = np.asarray(lc.quality) == 0
    return CentroidSeries(
        np.asarray(lc.time.value, dtype=float)[good],
        np.asarray(lc.centroid_col.value, dtype=float)[good],
        np.asarray(lc.centroid_row.value, dtype=float)[good],
    )


def _cached_paths(search, download_dir: Path) -> list[Path]:
    """Where lightkurve caches each product (mirrors ``SearchResult._download_one``)."""
    table = search.table
    return [
        Path(
            download_dir,
            "mastDownload",
            row["obs_collection"],
            row["obs_id"],
            row["productFilename"],
        )
        for row in table
    ]


def detrend(lc, cand: Candidate, window_durations: float = 3.0):
    """Flatten stellar variability while masking the candidate's transits."""
    time = np.asarray(lc.time.value)
    cadence = float(np.nanmedian(np.diff(time)))
    window = max(int(window_durations * cand.duration / cadence), 11)
    window += 1 - window % 2  # savgol filter needs an odd window
    mask = in_transit_mask(time, cand, width=1.5)
    return lc.flatten(window_length=window, mask=mask)


def to_arrays(lc) -> tuple[np.ndarray, np.ndarray]:
    """Extract plain arrays of time and normalized flux."""
    time = np.asarray(lc.time.value, dtype=float)
    flux = np.asarray(lc.flux.value, dtype=float)
    good = np.isfinite(time) & np.isfinite(flux)
    return time[good], flux[good]


def load_detrended(
    cand: Candidate, author: str = "SPOC"
) -> tuple[np.ndarray, np.ndarray, list[CentroidSeries]]:
    """Detrended time and flux, plus each sector's centroids."""
    sectors = fetch_sectors(cand.tic_id, author=author)
    lc = sectors.stitch().remove_nans()
    time, flux = to_arrays(detrend(lc, cand))
    return time, flux, [centroid_series(s) for s in sectors]
