"""Download and detrend TESS light curves via lightkurve."""

from __future__ import annotations

import functools
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from exovet.candidate import Candidate
from exovet.diagnostics.centroid import CentroidSeries
from exovet.diagnostics.folding import in_transit_mask

DEFAULT_DOWNLOAD_DIR = Path("cache/lightcurves")
MAX_SECTORS = 10
# Pipelines to try in order. SPOC has 2-minute light curves with centroids; QLP
# covers many fainter targets from the full-frame images, without centroids.
DEFAULT_AUTHORS = ("SPOC",)
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


@dataclass(frozen=True)
class LightCurveData:
    """Detrended light curve of one target, and where it came from."""

    time: np.ndarray
    flux: np.ndarray
    centroids: list[CentroidSeries]
    author: str


def search_sectors(tic_id: int, author: str):
    """Search one pipeline's light curves for a target."""
    import lightkurve as lk

    enforce_http_timeout()
    # SPOC is restricted to 2-minute cadence so sectors stitch cleanly; other
    # pipelines take whatever cadence is available.
    exptime = 120 if author == "SPOC" else None
    return lk.search_lightcurve(f"TIC {tic_id}", mission="TESS", author=author, exptime=exptime)


def fetch_sectors(
    tic_id: int,
    authors: Sequence[str] = DEFAULT_AUTHORS,
    download_dir: Path = DEFAULT_DOWNLOAD_DIR,
    max_sectors: int | None = MAX_SECTORS,
):
    """Download up to ``max_sectors`` sectors for a target, one light curve each.

    Pipelines in ``authors`` are tried in order and the first with data wins.
    The sector cap keeps targets in the continuous viewing zone (~40 sectors)
    tractable.
    """
    import lightkurve as lk

    for author in authors:
        search = search_sectors(tic_id, author)
        if len(search):
            break
    else:
        raise LookupError(f"No {'/'.join(authors)} light curves for TIC {tic_id}")
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


def _values(column) -> np.ndarray:
    return np.asarray(getattr(column, "value", column), dtype=float)


def centroid_series(lc) -> CentroidSeries | None:
    """Flux-weighted centroids of one sector, keeping only unflagged cadences.

    None when the pipeline provides no usable centroids, as QLP does not.
    """
    if any(getattr(lc, name, None) is None for name in ("centroid_col", "centroid_row")):
        return None
    col, row = _values(lc.centroid_col), _values(lc.centroid_row)
    if np.all(np.isnan(col)) or np.all(np.isnan(row)):
        return None
    good = np.asarray(lc.quality) == 0
    return CentroidSeries(_values(lc.time)[good], col[good], row[good])


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


def load_detrended(cand: Candidate, authors: Sequence[str] = DEFAULT_AUTHORS) -> LightCurveData:
    """Detrended light curve of a candidate's target, centroids where available."""
    sectors = fetch_sectors(cand.tic_id, authors=authors)
    lc = sectors.stitch().remove_nans()
    time, flux = to_arrays(detrend(lc, cand))
    centroids = (centroid_series(s) for s in sectors)
    author = str(sectors[0].meta.get("AUTHOR", authors[0]))
    return LightCurveData(time, flux, [c for c in centroids if c is not None], author)
