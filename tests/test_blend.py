"""Tests for crossfade layer opacity math."""

from __future__ import annotations

from cellacdc.blend import crossfade_opacities, layer_opacities


def test_crossfade_opacities_endpoints() -> None:
    assert crossfade_opacities(0) == (1.0, 0.0)
    assert crossfade_opacities(100) == (0.0, 1.0)
    assert crossfade_opacities(50) == (0.5, 0.5)


def test_layer_opacities_without_secondary() -> None:
    primary, secondary, seg = layer_opacities(50, 0, has_secondary=False)
    assert secondary == 0.0
    assert primary == 1.0
    assert seg == 0.0

    primary, secondary, seg = layer_opacities(50, 100, has_secondary=False)
    assert primary == 0.0
    assert secondary == 0.0
    assert seg == 1.0


def test_layer_opacities_with_secondary() -> None:
    primary, secondary, seg = layer_opacities(0, 0, has_secondary=True)
    assert primary == 1.0
    assert secondary == 0.0
    assert seg == 0.0

    primary, secondary, seg = layer_opacities(100, 50, has_secondary=True)
    assert primary == 0.0
    assert secondary == 0.5
    assert seg == 0.5
