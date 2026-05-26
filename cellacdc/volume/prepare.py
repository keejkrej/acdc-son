"""Extract and normalize volumes for 3D rendering."""

from __future__ import annotations

import numpy as np

from cellacdc.data import ImagedData, SegmentationResult
from cellacdc.segmentation import tools


def volume_zyx(
    imaged: ImagedData,
    *,
    t_index: int = 0,
) -> np.ndarray:
    """Return a ``(Z, Y, X)`` stack for the given time index."""
    vol4 = tools.normalize_to_4d(imaged.image, imaged.layout)
    t = max(0, min(int(t_index), vol4.shape[0] - 1))
    return np.asarray(vol4[t])


def mask_volume_zyx(
    result: SegmentationResult,
    layout: tools.StackLayout,
    *,
    t_index: int = 0,
) -> np.ndarray:
    """Return label IDs as ``(Z, Y, X)`` for the given time index."""
    vol4 = tools.normalize_to_4d(result.mask, layout)
    t = max(0, min(int(t_index), vol4.shape[0] - 1))
    return np.asarray(vol4[t])


def normalize_image_volume(volume: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    """Scale a volume to ``float32`` in ``[0, 1]`` and return original clim."""
    vol = np.asarray(volume, dtype=np.float32)
    vmin = float(vol.min())
    vmax = float(vol.max())
    if vmax <= vmin:
        vmax = vmin + 1.0
    scaled = (vol - vmin) / (vmax - vmin)
    return np.ascontiguousarray(scaled), (0.0, 1.0)


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
