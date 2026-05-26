"""Intensity level helpers for microscopy display."""

from __future__ import annotations

import numpy as np

from acdc.core.stack import StackShape, normalize_to_4d


def autoscale_levels(data: np.ndarray) -> tuple[float, float]:
    """Return ``(lo, hi)`` display levels using global min and max."""
    arr = np.asarray(data)
    if arr.size == 0:
        return 0.0, 1.0
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0, 1.0
    lo = float(finite.min())
    hi = float(finite.max())
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def stack_autoscale_levels(
    data: np.ndarray,
    stack_shape: StackShape,
) -> tuple[float, float]:
    """Autoscale across every frame and Z slice in a stack (min/max)."""
    vol4 = normalize_to_4d(data, stack_shape)
    return autoscale_levels(vol4)


def stack_display_levels(
    data: np.ndarray,
    stack_shape: StackShape,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return stack window ``(lo, hi)`` and default normalized clim (once per load)."""
    stack_lo, stack_hi = stack_autoscale_levels(data, stack_shape)
    vol4 = normalize_to_4d(data, stack_shape)
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
