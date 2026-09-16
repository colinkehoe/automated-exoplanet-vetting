import numpy as np
import pytest

from exovet.candidate import Candidate
from exovet.diagnostics.folding import phase_offset, transit_number

NOISE = 1e-4


def synthetic_lightcurve(
    cand: Candidate,
    odd_depth: float | None = None,
    secondary_depth: float = 0.0,
    v_shaped: bool = False,
    days: float = 54.0,
    cadence: float = 2 / 1440,
    seed: int = 0,
):
    """Flat light curve with box (or triangular) transits plus Gaussian noise."""
    rng = np.random.default_rng(seed)
    time = np.arange(0.0, days, cadence)
    flux = 1.0 + rng.normal(0, NOISE, time.size)

    offset = np.abs(phase_offset(time, cand.period, cand.epoch))
    in_transit = offset < 0.5 * cand.duration
    depth = np.full(time.size, cand.depth)
    if odd_depth is not None:
        depth[transit_number(time, cand.period, cand.epoch) % 2 == 1] = odd_depth
    profile = 1 - offset / (0.5 * cand.duration) if v_shaped else 1.0
    # Triangle with the same mean depth as the box.
    scale = 2.0 if v_shaped else 1.0
    flux -= np.where(in_transit, scale * depth * profile, 0.0)

    if secondary_depth:
        sec = np.abs(phase_offset(time, cand.period, cand.epoch + 0.5 * cand.period))
        flux -= np.where(sec < 0.5 * cand.duration, secondary_depth, 0.0)
    return time, flux


@pytest.fixture
def cand():
    return Candidate(tic_id=1, period=3.0, epoch=1.0, duration=0.125, depth=2e-3)
