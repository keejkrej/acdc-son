"""Crossfade math for primary, secondary, and segmentation layers."""

from __future__ import annotations


def crossfade_opacities(value_0_to_100: float) -> tuple[float, float]:
    """Return ``(first, second)`` opacities for a 0–100 crossfade slider."""
    t = max(0.0, min(100.0, float(value_0_to_100))) / 100.0
    return 1.0 - t, t


def layer_opacities(
    primary_secondary_blend_0_to_100: float,
    image_seg_blend_0_to_100: float,
    *,
    has_secondary: bool,
) -> tuple[float, float, float]:
    """Return ``(primary_opacity, secondary_opacity, seg_opacity)`` for display layers."""
    if has_secondary:
        primary_w, secondary_w = crossfade_opacities(primary_secondary_blend_0_to_100)
    else:
        primary_w, secondary_w = 1.0, 0.0
    image_scale, seg_scale = crossfade_opacities(image_seg_blend_0_to_100)
    return primary_w * image_scale, secondary_w * image_scale, seg_scale
