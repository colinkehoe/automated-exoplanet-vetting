"""Consistency of the transit with its host star.

A planet transiting the catalogued star has a duration set by the star's
density and the orbital period, and a radius that is only as large as the
depth and stellar radius allow. Eclipsing binaries and signals blended from a
different star often fail one of these checks.
"""

from __future__ import annotations

import math

from exovet.candidate import Candidate, Star

G_CGS = 6.674e-8
R_SUN_CM = 6.957e10
M_SUN_G = 1.989e33
R_SUN_IN_R_EARTH = 109.08
SECONDS_PER_DAY = 86400.0

# Features built from catalog stellar values that follow-up tends to fill in:
# confirmed planets almost never lack them, while unresolved candidates and
# false positives often do. Missingness would therefore leak the label, so the
# model imputes these instead of letting the trees route NaN.
BACKFILLED_FEATURES = [
    "star_teff",
    "star_logg",
    "star_radius",
    "star_log_density",
    "star_distance",
    "log_planet_radius",
    "log_duration_ratio",
]


def stellar_mass(star: Star) -> float:
    """Catalog mass, falling back to one derived from log g and radius."""
    if math.isfinite(star.mass) and star.mass > 0:
        return star.mass
    if math.isfinite(star.logg) and math.isfinite(star.radius):
        return 10**star.logg * (star.radius * R_SUN_CM) ** 2 / G_CGS / M_SUN_G
    return math.nan


def expected_duration(period: float, depth: float, radius: float, mass: float) -> float:
    """Duration (days) of a central transit on a circular orbit."""
    if not (radius > 0 and mass > 0):  # also False for NaN
        return math.nan
    period_s = period * SECONDS_PER_DAY
    a = (G_CGS * mass * M_SUN_G * period_s**2 / (4 * math.pi**2)) ** (1 / 3)
    k = math.sqrt(depth) if depth > 0 else 0.0
    chord = radius * R_SUN_CM * (1 + k) / a
    if chord >= 1:  # contact binary territory: the "orbit" is inside the star
        return period / 2
    return period / math.pi * math.asin(chord)


def stellar_features(cand: Candidate) -> dict[str, float]:
    star = cand.star
    mass = stellar_mass(star)
    density = mass / star.radius**3 if star.radius > 0 else math.nan
    t_exp = expected_duration(cand.period, cand.depth, star.radius, mass)
    planet_radius = (
        math.sqrt(cand.depth) * star.radius * R_SUN_IN_R_EARTH if cand.depth > 0 else math.nan
    )

    def log10(x):
        return math.log10(x) if x > 0 else math.nan  # also NaN for NaN

    return {
        "star_teff": star.teff,
        "star_logg": star.logg,
        "star_radius": star.radius,
        "star_log_density": log10(density),
        "star_distance": star.distance,
        "tess_mag": star.tess_mag,
        "log_planet_radius": log10(planet_radius),
        "log_duration_ratio": log10(cand.duration / t_exp),
    }
