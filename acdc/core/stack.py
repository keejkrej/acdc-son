"""Stack dimension shape and slice helpers for 2D–4D microscopy volumes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StackShape:
    """Normalized view of image/mask dimensionality."""

    ndim: int
    has_time: bool
    has_z: bool
    size_t: int
    size_z: int
    size_y: int
    size_x: int


def infer_shape(array_shape: tuple[int, ...]) -> StackShape:
    if len(array_shape) == 2:
        y, x = array_shape
        return StackShape(2, False, False, 1, 1, y, x)
    if len(array_shape) == 3:
        d0, y, x = array_shape
        if d0 <= 64:
            return StackShape(3, False, True, 1, d0, y, x)
        return StackShape(3, True, False, d0, 1, y, x)
    if len(array_shape) == 4:
        t, z, y, x = array_shape
        return StackShape(4, True, True, t, z, y, x)
    raise ValueError(f"Expected 2–4 dimensions, got {array_shape}")


def shape_from_metadata(
    array_shape: tuple[int, ...],
    size_t: int | None,
    size_z: int | None,
) -> StackShape:
    """Build stack shape using metadata ``SizeT``/``SizeZ`` when available."""
    if size_t is None or size_z is None:
        return infer_shape(array_shape)

    yx = array_shape[-2:]
    if len(array_shape) == 2:
        return StackShape(2, False, False, max(1, size_t), max(1, size_z), yx[0], yx[1])
    if len(array_shape) == 3:
        d0 = array_shape[0]
        has_time = size_t > 1 and size_z <= 1
        has_z = size_z > 1 and size_t <= 1
        if has_time:
            return StackShape(3, True, False, d0, 1, yx[0], yx[1])
        if has_z:
            return StackShape(3, False, True, 1, d0, yx[0], yx[1])
        if size_t > 1:
            return StackShape(3, True, False, d0, 1, yx[0], yx[1])
        return StackShape(3, False, True, 1, d0, yx[0], yx[1])
    if len(array_shape) == 4:
        t, z = array_shape[0], array_shape[1]
        return StackShape(4, True, True, t, z, yx[0], yx[1])
    raise ValueError(f"Expected 2–4 dimensions, got {array_shape}")


def normalize_to_4d(data: np.ndarray, stack_shape: StackShape) -> np.ndarray:
    """Return array shaped (T, Z, Y, X)."""
    arr = np.asarray(data)
    if stack_shape.ndim == 2:
        return arr.reshape(1, 1, stack_shape.size_y, stack_shape.size_x)
    if stack_shape.ndim == 3:
        if stack_shape.has_z:
            return arr.reshape(1, stack_shape.size_z, stack_shape.size_y, stack_shape.size_x)
        return arr.reshape(stack_shape.size_t, 1, stack_shape.size_y, stack_shape.size_x)
    return arr


def extract_slice(volume: np.ndarray, stack_shape: StackShape, t: int, z: int) -> np.ndarray:
    """Return a 2D (Y, X) slice from 2D–4D data."""
    vol4 = normalize_to_4d(volume, stack_shape)
    t = max(0, min(t, vol4.shape[0] - 1))
    z = max(0, min(z, vol4.shape[1] - 1))
    return vol4[t, z]


def write_slice(
    volume: np.ndarray,
    stack_shape: StackShape,
    t: int,
    z: int,
    slice_2d: np.ndarray,
) -> np.ndarray:
    """Write a 2D slice back into a copy of ``volume``."""
    out = np.array(volume, copy=True)
    vol4 = normalize_to_4d(out, stack_shape)
    t = max(0, min(t, vol4.shape[0] - 1))
    z = max(0, min(z, vol4.shape[1] - 1))
    vol4[t, z] = slice_2d
    if stack_shape.ndim == 2:
        return vol4[0, 0]
    if stack_shape.ndim == 3:
        if stack_shape.has_z:
            return vol4[0]
        return vol4[:, 0]
    return vol4
