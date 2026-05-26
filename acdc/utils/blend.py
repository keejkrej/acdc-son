"""Layer opacity math for multi-channel images and segmentation."""

from __future__ import annotations

from collections.abc import Sequence


def display_opacities(
    channel_weights_0_to_1: Sequence[float],
    image_seg_blend_0_to_100: float,
) -> tuple[list[float], float]:
    """Return per-channel opacities and segmentation opacity for the viewer."""
    weights = [max(0.0, min(1.0, float(w))) for w in channel_weights_0_to_1]
    total = sum(weights)
    if total <= 0.0:
        n = len(weights)
        shares = [0.0] * n if n == 0 else [1.0 / n] * n
    else:
        shares = [w / total for w in weights]

    t = max(0.0, min(100.0, float(image_seg_blend_0_to_100))) / 100.0
    image_scale = 1.0 - t
    return [share * image_scale for share in shares], t
