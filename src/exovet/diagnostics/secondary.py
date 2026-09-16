"""Secondary eclipse search.

A significant dip away from the primary transit points to an eclipsing binary
(or a very hot, self-luminous companion).
"""

from __future__ import annotations

import numpy as np

from exovet.candidate import Candidate
from exovet.diagnostics.folding import measure_depth, phase_offset


def _depth_at_phase(time, flux, cand, phase, out_mask):
    offset = phase_offset(time, cand.period, cand.epoch + phase * cand.period)
    return measure_depth(flux, np.abs(offset) < 0.5 * cand.duration, out_mask)


def secondary_features(
    time: np.ndarray, flux: np.ndarray, cand: Candidate, n_phases: int = 100
) -> dict[str, float]:
    offset = phase_offset(time, cand.period, cand.epoch)
    # Exclude the primary from the baseline and from the scan.
    far_from_primary = np.abs(offset) > 1.5 * cand.duration
    primary = measure_depth(flux, np.abs(offset) < 0.5 * cand.duration, far_from_primary)

    at_half = _depth_at_phase(time, flux, cand, 0.5, far_from_primary)

    # Scan phases that keep the test window clear of the primary, to catch
    # secondaries shifted by eccentricity.
    margin = 1.5 * cand.duration / cand.period
    best = at_half
    for phase in np.linspace(margin, 1 - margin, n_phases):
        d = _depth_at_phase(time, flux, cand, phase, far_from_primary)
        if np.isfinite(d.snr) and (not np.isfinite(best.snr) or d.snr > best.snr):
            best = d

    def ratio(d):
        return d.value / primary.value if primary.value > 0 else float("nan")

    return {
        "secondary_snr": at_half.snr,
        "secondary_ratio": ratio(at_half),
        "secondary_max_snr": best.snr,
        "secondary_max_ratio": ratio(best),
    }
