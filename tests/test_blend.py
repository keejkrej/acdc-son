"""Tests for multi-channel layer opacity math."""

from __future__ import annotations

import pytest

from acdc.blend import crossfade_opacities, display_opacities, normalized_weights


def test_crossfade_opacities_endpoints() -> None:
    assert crossfade_opacities(0) == (1.0, 0.0)
    assert crossfade_opacities(100) == (0.0, 1.0)
    assert crossfade_opacities(50) == (0.5, 0.5)


def test_normalized_weights_divides_by_sum() -> None:
    assert normalized_weights([1.0, 1.0, 1.0]) == pytest.approx([1 / 3, 1 / 3, 1 / 3])
    assert normalized_weights([1.0, 0.5]) == pytest.approx([2 / 3, 1 / 3])


def test_normalized_weights_all_zero_splits_evenly() -> None:
    assert normalized_weights([0.0, 0.0]) == [0.5, 0.5]


def test_display_opacities_scales_channels_and_seg() -> None:
    channels, seg = display_opacities([1.0, 1.0], 0)
    assert channels == pytest.approx([0.5, 0.5])
    assert seg == 0.0

    channels, seg = display_opacities([2.0, 1.0], 100)
    assert channels == pytest.approx([0.0, 0.0])
    assert seg == 1.0

    channels, seg = display_opacities([1.0, 0.0], 50)
    assert channels == pytest.approx([0.5, 0.0])
    assert seg == pytest.approx(0.5)
