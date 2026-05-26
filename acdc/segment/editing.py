"""Mask editing and label overlay helpers for 2D segmentation."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


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
