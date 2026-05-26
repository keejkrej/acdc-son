"""Tests for display level autoscaling."""

from __future__ import annotations

import numpy as np

from cellacdc.display_levels import autoscale_levels, scale_to_unit, stack_autoscale_levels
from cellacdc.segmentation import tools


def test_autoscale_levels_stretches_sparse_signal() -> None:
    data = np.zeros((16, 16), dtype=np.uint16)
    data[4:8, 4:8] = 500
    lo, hi = autoscale_levels(data)
    assert lo < 500
    assert hi >= 500


def test_autoscale_levels_handles_flat_image() -> None:
    data = np.full((8, 8), 42, dtype=np.uint8)
    lo, hi = autoscale_levels(data)
    assert hi > lo


def test_stack_autoscale_ignores_hot_outlier_frame() -> None:
    stack = np.zeros((8, 8, 8), dtype=np.uint16)
    stack[2, 1:7, 1:7] = 120
    stack[4, 4, 4] = 60000
    layout = tools.infer_layout(stack.shape)
    lo, hi = stack_autoscale_levels(stack, layout)
    scaled = scale_to_unit(stack[2, 1:7, 1:7], lo, hi)
    assert hi < 60000
    assert scaled[0, 0] > 0.4


def test_scale_to_unit_clips_outside_range() -> None:
    data = np.array([0, 50, 100], dtype=np.float32)
    scaled = scale_to_unit(data, 25.0, 75.0)
    assert scaled.tolist() == [0.0, 0.5, 1.0]
