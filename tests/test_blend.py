"""Tests for multi-channel layer opacity math."""

from __future__ import annotations

import pytest

from acdc.utils.blend import display_opacities


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

    channels, seg = display_opacities([0.0, 0.0], 0)
    assert channels == pytest.approx([0.5, 0.5])
    assert seg == 0.0

    channels, seg = display_opacities([1.0, 1.0, 1.0], 0)
    assert channels == pytest.approx([1 / 3, 1 / 3, 1 / 3])
    assert seg == 0.0

    channels, seg = display_opacities([1.0, 0.5], 0)
    assert channels == pytest.approx([2 / 3, 1 / 3])
    assert seg == 0.0
