"""Brush/eraser operations and stack dimension helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class StackLayout:
    """Normalized view of image/mask dimensionality."""

    ndim: int
    has_time: bool
    has_z: bool
    size_t: int
    size_z: int
    size_y: int
    size_x: int


def infer_layout(shape: tuple[int, ...]) -> StackLayout:
    if len(shape) == 2:
        y, x = shape
        return StackLayout(2, False, False, 1, 1, y, x)
    if len(shape) == 3:
        # Heuristic: if first dim is small (<=64), treat as Z; else T
        d0, y, x = shape
        if d0 <= 64:
            return StackLayout(3, False, True, 1, d0, y, x)
        return StackLayout(3, True, False, d0, 1, y, x)
    if len(shape) == 4:
        t, z, y, x = shape
        return StackLayout(4, True, True, t, z, y, x)
    raise ValueError(f"Expected 2–4 dimensions, got {shape}")


def layout_from_metadata(
    shape: tuple[int, ...],
    size_t: int | None,
    size_z: int | None,
) -> StackLayout:
    """Build stack layout using metadata ``SizeT``/``SizeZ`` when available."""
    if size_t is None or size_z is None:
        return infer_layout(shape)

    yx = shape[-2:]
    if len(shape) == 2:
        return StackLayout(2, False, False, max(1, size_t), max(1, size_z), yx[0], yx[1])
    if len(shape) == 3:
        d0 = shape[0]
        has_time = size_t > 1 and size_z <= 1
        has_z = size_z > 1 and size_t <= 1
        if has_time:
            return StackLayout(3, True, False, d0, 1, yx[0], yx[1])
        if has_z:
            return StackLayout(3, False, True, 1, d0, yx[0], yx[1])
        if size_t > 1:
            return StackLayout(3, True, False, d0, 1, yx[0], yx[1])
        return StackLayout(3, False, True, 1, d0, yx[0], yx[1])
    if len(shape) == 4:
        t, z = shape[0], shape[1]
        return StackLayout(4, True, True, t, z, yx[0], yx[1])
    raise ValueError(f"Expected 2–4 dimensions, got {shape}")


def normalize_to_4d(data: np.ndarray, layout: StackLayout) -> np.ndarray:
    """Return array shaped (T, Z, Y, X)."""
    arr = np.asarray(data)
    if layout.ndim == 2:
        return arr.reshape(1, 1, layout.size_y, layout.size_x)
    if layout.ndim == 3:
        if layout.has_z:
            return arr.reshape(1, layout.size_z, layout.size_y, layout.size_x)
        return arr.reshape(layout.size_t, 1, layout.size_y, layout.size_x)
    return arr


def extract_slice(volume: np.ndarray, layout: StackLayout, t: int, z: int) -> np.ndarray:
    """Return a 2D (Y, X) slice from 2D–4D data."""
    vol4 = normalize_to_4d(volume, layout)
    t = max(0, min(t, vol4.shape[0] - 1))
    z = max(0, min(z, vol4.shape[1] - 1))
    return vol4[t, z]


def write_slice(
    volume: np.ndarray,
    layout: StackLayout,
    t: int,
    z: int,
    slice_2d: np.ndarray,
) -> np.ndarray:
    """Write a 2D slice back into a copy of ``volume``."""
    out = np.array(volume, copy=True)
    vol4 = normalize_to_4d(out, layout)
    t = max(0, min(t, vol4.shape[0] - 1))
    z = max(0, min(z, vol4.shape[1] - 1))
    vol4[t, z] = slice_2d
    if layout.ndim == 2:
        return vol4[0, 0]
    if layout.ndim == 3:
        if layout.has_z:
            return vol4[0]
        return vol4[:, 0]
    return vol4


def apply_brush(
    mask_slice: np.ndarray,
    y: int,
    x: int,
    radius: int,
    label: int,
    *,
    erase: bool = False,
) -> None:
    """Paint or erase a circular region in-place on a 2D mask slice."""
    h, w = mask_slice.shape
    r = max(1, int(radius))
    y0, y1 = max(0, y - r), min(h, y + r + 1)
    x0, x1 = max(0, x - r), min(w, x + r + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    dist2 = (yy - y) ** 2 + (xx - x) ** 2
    circle = dist2 <= r * r
    if erase:
        mask_slice[y0:y1, x0:x1][circle] = 0
    else:
        mask_slice[y0:y1, x0:x1][circle] = np.uint32(label)


def apply_brush_stroke(
    mask_slice: np.ndarray,
    y: int,
    x: int,
    prev_y: int | None,
    prev_x: int | None,
    radius: int,
    label: int,
    *,
    erase: bool = False,
) -> None:
    """Paint a continuous stroke by interpolating disks between successive points."""
    if prev_y is None or prev_x is None:
        apply_brush(mask_slice, y, x, radius, label, erase=erase)
        return
    steps = max(abs(y - prev_y), abs(x - prev_x))
    if steps == 0:
        apply_brush(mask_slice, y, x, radius, label, erase=erase)
        return
    ys = np.linspace(prev_y, y, steps + 1, dtype=int)
    xs = np.linspace(prev_x, x, steps + 1, dtype=int)
    for py, px in zip(ys, xs, strict=False):
        apply_brush(mask_slice, int(py), int(px), radius, label, erase=erase)


def fill_label_holes(mask_slice: np.ndarray, label: int) -> bool:
    """Fill interior holes in the region with ``label``. Returns True if modified."""
    if label <= 0:
        return False
    region = mask_slice == label
    if not np.any(region):
        return False
    filled = ndimage.binary_fill_holes(region)
    if not np.any(filled & ~region):
        return False
    mask_slice[filled] = np.uint32(label)
    return True


def filter_visible_labels(
    mask_slice: np.ndarray,
    visible_ids: set[int],
) -> np.ndarray:
    """Return a mask slice showing only ``visible_ids`` (background elsewhere)."""
    if not visible_ids:
        return np.zeros_like(mask_slice)
    visible = np.fromiter(visible_ids, dtype=mask_slice.dtype)
    return np.where(np.isin(mask_slice, visible), mask_slice, np.uint32(0))


def apply_label_visibility(
    mask_slice: np.ndarray,
    hidden_ids: set[int],
) -> np.ndarray:
    """Return ``mask_slice`` with ``hidden_ids`` zeroed out."""
    if not hidden_ids:
        return mask_slice
    hidden = np.fromiter(hidden_ids, dtype=mask_slice.dtype)
    return np.where(np.isin(mask_slice, hidden), np.uint32(0), mask_slice)


def find_outer_boundaries(label_img: np.ndarray) -> np.ndarray:
    """Return a bool mask of outer label edges (4-connected)."""
    lab = np.asarray(label_img)
    fg = lab > 0
    if not np.any(fg):
        return np.zeros(lab.shape, dtype=bool)
    boundary = np.zeros(lab.shape, dtype=bool)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        neighbor = np.zeros_like(lab)
        if dy == -1:
            neighbor[1:, :] = lab[:-1, :]
        elif dy == 1:
            neighbor[:-1, :] = lab[1:, :]
        elif dx == -1:
            neighbor[:, 1:] = lab[:, :-1]
        else:
            neighbor[:, :-1] = lab[:, 1:]
        boundary |= fg & (neighbor != lab)
    return boundary


def normalize_rect(
    y0: int,
    x0: int,
    y1: int,
    x1: int,
) -> tuple[int, int, int, int]:
    """Return inclusive ``(ymin, xmin, ymax, xmax)`` from two corners."""
    return min(y0, y1), min(x0, x1), max(y0, y1), max(x0, x1)


def unique_labels_in_rect(
    mask_slice: np.ndarray,
    y0: int,
    x0: int,
    y1: int,
    x1: int,
) -> list[int]:
    """Return sorted label IDs fully contained in the inclusive rectangle."""
    ymin, xmin, ymax, xmax = normalize_rect(y0, x0, y1, x1)
    h, w = mask_slice.shape
    ymin = max(0, ymin)
    xmin = max(0, xmin)
    ymax = min(h - 1, ymax)
    xmax = min(w - 1, xmax)
    if ymin > ymax or xmin > xmax:
        return []
    region = mask_slice[ymin : ymax + 1, xmin : xmax + 1]
    contained: list[int] = []
    for label in np.unique(region):
        label_id = int(label)
        if label_id <= 0:
            continue
        ys, xs = np.where(mask_slice == label_id)
        if (
            ys.min() >= ymin
            and ys.max() <= ymax
            and xs.min() >= xmin
            and xs.max() <= xmax
        ):
            contained.append(label_id)
    return sorted(contained)


def label_bounding_box(
    mask_slice: np.ndarray,
    label_id: int,
) -> tuple[int, int, int, int] | None:
    """Return inclusive ``(ymin, xmin, ymax, xmax)`` for ``label_id``."""
    if label_id <= 0:
        return None
    ys, xs = np.where(mask_slice == label_id)
    if ys.size == 0:
        return None
    return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())


def labels_to_contour_rgba(
    mask_slice: np.ndarray,
    lut: np.ndarray,
    *,
    thickness: int = 1,
    alpha: int = 255,
) -> np.ndarray:
    """Build an RGBA contour overlay for visible labels."""
    h, w = mask_slice.shape
    rgba = np.zeros((h, w, 4), dtype=np.ubyte)
    boundary = find_outer_boundaries(mask_slice)
    if not np.any(boundary):
        return rgba
    if thickness > 1:
        boundary = ndimage.binary_dilation(boundary, iterations=thickness - 1)
    ids = mask_slice[boundary].astype(np.intp)
    rgba[boundary] = lut[ids]
    rgba[boundary, 3] = np.ubyte(alpha)
    return rgba


def _label_rgb(label: int, c: float = 0.85) -> tuple[float, float, float]:
    hue = (int(label) * 47) % 360
    hp = hue / 60.0
    x = c * (1 - abs(hp % 2 - 1))
    if hp < 1:
        return (c, x, 0)
    if hp < 2:
        return (x, c, 0)
    if hp < 3:
        return (0, c, x)
    if hp < 4:
        return (0, x, c)
    if hp < 5:
        return (x, 0, c)
    return (c, 0, x)


def build_label_lut(num_entries: int = 4096, alpha: float = 0.45) -> np.ndarray:
    """Build an RGBA lookup table for label IDs (index 0 is transparent)."""
    lut = np.zeros((num_entries, 4), dtype=np.ubyte)
    alpha_u8 = int(round(alpha * 255))
    for label in range(1, num_entries):
        r, g, b = _label_rgb(label)
        lut[label] = (int(r * 255), int(g * 255), int(b * 255), alpha_u8)
    return lut


def labels_to_rgba(mask_slice: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Map label IDs to RGBA overlay (background transparent)."""
    h, w = mask_slice.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    labels = np.unique(mask_slice)
    labels = labels[labels > 0]
    if labels.size == 0:
        return rgba
    for label in labels:
        sel = mask_slice == label
        r, g, b = _label_rgb(label)
        rgba[sel, 0] = r
        rgba[sel, 1] = g
        rgba[sel, 2] = b
        rgba[sel, 3] = alpha
    return rgba
