"""2D segmentation viewer middleware."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from acdc.core.data import AcdcData, AcdcResult, coalesce_images
from acdc.middleware.context import AcdcContext
from acdc.segment.viewer import run_segment as open_segment


def run_segment(ctx: AcdcContext, next_: Callable[[], None]) -> None:
    """Open the 2D editor; block until the window closes."""
    ctx.images, _ = open_segment(ctx.images, ctx.segmentation)
    next_()


def from_arrays(
    images: Sequence[AcdcData],
    segmentation: AcdcResult,
) -> AcdcContext:
    """Build a context from pre-loaded data."""
    return AcdcContext(images=coalesce_images(images), segmentation=segmentation)
