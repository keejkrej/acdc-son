"""Tests for display level autoscaling."""

from __future__ import annotations

import numpy as np

from acdc.utils.display_levels import autoscale_levels, scale_to_unit, stack_autoscale_levels
from acdc.core import stack


def test_autoscale_levels_uses_min_max() -> None:
    data = np.zeros((16, 16), dtype=np.uint16)
    data[4:8, 4:8] = 500
    lo, hi = autoscale_levels(data)
    assert lo == 0.0
    assert hi == 500.0


def test_autoscale_levels_handles_flat_image() -> None:
    data = np.full((8, 8), 42, dtype=np.uint8)
    lo, hi = autoscale_levels(data)
    assert hi > lo


def test_stack_autoscale_uses_global_min_max() -> None:
    volume = np.zeros((8, 8, 8), dtype=np.uint16)
    volume[2, 1:7, 1:7] = 120
    volume[4, 4, 4] = 60000
    stack_shape = stack.infer_shape(volume.shape)
    lo, hi = stack_autoscale_levels(volume, stack_shape)
    assert lo == 0.0
    assert hi == 60000.0
    scaled = scale_to_unit(volume[2, 1:7, 1:7], lo, hi)
    assert scaled[0, 0] == 120 / 60000


def test_scale_to_unit_clips_outside_range() -> None:
    data = np.array([0, 50, 100], dtype=np.float32)
    scaled = scale_to_unit(data, 25.0, 75.0)
    assert scaled.tolist() == [0.0, 0.5, 1.0]
