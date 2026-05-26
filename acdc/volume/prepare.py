"""Extract and normalize volumes for 3D rendering."""

from __future__ import annotations

import numpy as np

from acdc.core.data import AcdcData, AcdcResult
from acdc.utils.display_levels import autoscale_levels, scale_to_unit, stack_display_levels
from acdc.core import stack


def volume_zyx(
    imaged: AcdcData,
    *,
    t_index: int = 0,
) -> np.ndarray:
    """Return a ``(Z, Y, X)`` stack for the given time index."""
    vol4 = stack.normalize_to_4d(imaged.image, imaged.stack_shape)
    t = max(0, min(int(t_index), vol4.shape[0] - 1))
    return np.asarray(vol4[t])


def mask_volume_zyx(
    result: AcdcResult,
    stack_shape: stack.StackShape,
    *,
    t_index: int = 0,
) -> np.ndarray:
    """Return label IDs as ``(Z, Y, X)`` for the given time index."""
    vol4 = stack.normalize_to_4d(result.mask, stack_shape)
    t = max(0, min(int(t_index), vol4.shape[0] - 1))
    return np.asarray(vol4[t])


def normalize_image_volume(
    volume: np.ndarray,
    *,
    stack_lo: float | None = None,
    stack_hi: float | None = None,
    display_clim: tuple[float, float] | None = None,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Scale a volume to ``float32`` in ``[0, 1]`` using precomputed stack levels."""
    vol = np.asarray(volume, dtype=np.float32)
    if stack_lo is None or stack_hi is None:
        stack_lo, stack_hi = autoscale_levels(vol)
    scaled = scale_to_unit(vol, stack_lo, stack_hi)
    clim = display_clim if display_clim is not None else autoscale_levels(scaled)
    return np.ascontiguousarray(scaled), clim


def normalize_image_stack_volume(
    volume: np.ndarray,
    full_stack: np.ndarray,
    stack_shape: stack.StackShape,
    *,
    stack_levels: tuple[float, float] | None = None,
    display_clim: tuple[float, float] | None = None,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Normalize one volume using stack levels computed once at load."""
    if stack_levels is None or display_clim is None:
        stack_levels, display_clim = stack_display_levels(full_stack, stack_shape)
    return normalize_image_volume(
        volume,
        stack_lo=stack_levels[0],
        stack_hi=stack_levels[1],
        display_clim=display_clim,
    )


def voxel_display_scale(dz: float, dy: float, dx: float) -> tuple[float, float, float]:
    """Return vispy ``STTransform`` scale for ``(Z, Y, X)`` data (Cell-ACDC style)."""
    dx_eff = dx if dx > 0 else 1.0
    dy_eff = dy if dy > 0 else 1.0
    dz_eff = dz if dz > 0 else 1.0
    return (1.0, dy_eff / dx_eff, dz_eff / dx_eff)


def label_volume_for_vispy(volume: np.ndarray) -> tuple[np.ndarray, int]:
    """Return ``float32`` label IDs and a LUT sized to the actual label range."""
    lab = np.ascontiguousarray(volume, dtype=np.float32)
    max_label = int(lab.max(initial=0))
    lut_size = max(max_label + 1, 2)
    return lab, lut_size
