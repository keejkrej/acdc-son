"""Load experiment data into an :class:`AcdcContext`."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence

from acdc.data import SegmentationResult, load as load_data
from acdc.middleware.context import AcdcContext


def load(
    path: str | Path,
    *,
    channels: Sequence[str] | None = None,
    channel: str | None = None,
    position: str | None = None,
    segmentation: SegmentationResult | None = None,
) -> AcdcContext:
    """Load channel(s) and mask into a new context."""
    images, seg = load_data(
        path,
        channels=channels,
        channel=channel,
        position=position,
        segmentation=segmentation,
    )
    return AcdcContext(images=images, segmentation=seg)
