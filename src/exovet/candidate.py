"""Transit candidate representation."""

from __future__ import annotations

from dataclasses import dataclass

# TESS light curves use BTJD = BJD - 2457000.
BTJD_OFFSET = 2457000.0


NAN = float("nan")


@dataclass(frozen=True)
class Star:
    """Host star properties; NaN where the catalog has no value.

    Radius and mass in solar units, log g in cgs, distance in parsecs.
    """

    teff: float = NAN
    logg: float = NAN
    radius: float = NAN
    mass: float = NAN
    distance: float = NAN
    tess_mag: float = NAN


@dataclass(frozen=True)
class Candidate:
    """A periodic transit signal to be vetted.

    Times are in BTJD, durations and periods in days, depth as a fraction.
    """

    tic_id: int
    period: float
    epoch: float
    duration: float
    depth: float
    toi: str | None = None
    star: Star = Star()

    @property
    def name(self) -> str:
        return f"TOI-{self.toi}" if self.toi else f"TIC {self.tic_id}"
