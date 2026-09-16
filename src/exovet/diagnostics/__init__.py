"""Vetting diagnostics computed on detrended, phase-folded light curves.

Each diagnostic returns a flat ``dict[str, float]`` of features; NaN marks a
feature that could not be measured (e.g. no even transits observed).
"""
