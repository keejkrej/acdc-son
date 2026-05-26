"""Tests for napari-style volume GL blending helpers."""

from __future__ import annotations

from acdc.volume.gl_blend import volume_gl_state


def test_first_visible_additive_blends_to_black() -> None:
    state = volume_gl_state("additive", first_visible=True)
    assert state["blend_func"][:2] == ("src_alpha", "zero")


def test_additive_over_layer_uses_dst_alpha() -> None:
    state = volume_gl_state("additive", first_visible=False)
    assert state["blend_func"][:2] == ("src_alpha", "dst_alpha")


def test_translucent_no_depth_disables_depth_test() -> None:
    state = volume_gl_state("translucent_no_depth", first_visible=False)
    assert state["depth_test"] is False
