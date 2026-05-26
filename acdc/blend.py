"""Layer opacity math for multi-channel images and segmentation."""

from __future__ import annotations

from collections.abc import Sequence


def crossfade_opacities(value_0_to_100: float) -> tuple[float, float]:
    """Return ``(first, second)`` opacities for a 0–100 crossfade slider."""
    t = max(0.0, min(100.0, float(value_0_to_100))) / 100.0
    return 1.0 - t, t


def normalized_weights(raw: Sequence[float]) -> list[float]:
    """Map channel weights in ``[0, 1]`` to shares that sum to 1 (``a/(a+b+…)``)."""
    weights = [max(0.0, min(1.0, float(w))) for w in raw]
    total = sum(weights)
    if total <= 0.0:
        n = len(weights)
        return [0.0] * n if n == 0 else [1.0 / n] * n
    return [w / total for w in weights]


def display_opacities(
    channel_weights_0_to_1: Sequence[float],
    image_seg_blend_0_to_100: float,
) -> tuple[list[float], float]:
    """Return per-channel opacities and segmentation opacity for the viewer."""
    shares = normalized_weights(channel_weights_0_to_1)
    image_scale, seg_scale = crossfade_opacities(image_seg_blend_0_to_100)
    return [share * image_scale for share in shares], seg_scale
