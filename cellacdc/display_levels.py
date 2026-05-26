"""Intensity level helpers for microscopy display."""

from __future__ import annotations

import numpy as np

from cellacdc.segmentation import tools


def autoscale_levels(
    data: np.ndarray,
    *,
    low_percentile: float = 2.0,
    high_percentile: float = 98.0,
) -> tuple[float, float]:
    """Return ``(lo, hi)`` display levels for a grayscale slice or volume."""
    arr = np.asarray(data)
    if arr.size == 0:
        return 0.0, 1.0
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(finite, low_percentile))
    hi = float(np.percentile(finite, high_percentile))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def stack_autoscale_levels(
    data: np.ndarray,
    layout: tools.StackLayout,
    *,
    low_percentile: float = 2.0,
    high_percentile: float = 98.0,
) -> tuple[float, float]:
    """Autoscale across every frame and Z slice in a stack."""
    vol4 = tools.normalize_to_4d(data, layout)
    return autoscale_levels(
        vol4,
        low_percentile=low_percentile,
        high_percentile=high_percentile,
    )


def stack_display_levels(
    data: np.ndarray,
    layout: tools.StackLayout,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return stack window ``(lo, hi)`` and default normalized clim (once per load)."""
    stack_lo, stack_hi = stack_autoscale_levels(data, layout)
    vol4 = tools.normalize_to_4d(data, layout)
    scaled = scale_to_unit(vol4, stack_lo, stack_hi)
    clim = autoscale_levels(scaled)
    return (stack_lo, stack_hi), clim


def scale_to_unit(data: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Linearly map ``[lo, hi]`` to ``[0, 1]`` with clipping."""
    span = float(hi) - float(lo)
    if span <= 0:
        span = 1.0
    scaled = (np.asarray(data, dtype=np.float32) - float(lo)) / span
    return np.clip(scaled, 0.0, 1.0)
