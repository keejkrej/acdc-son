"""2D segmentation viewer middleware."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from acdc.data import ImageData, SegmentationResult, coalesce_images
from acdc.middleware.context import AcdcContext
from acdc.segment.viewer import run_segment


def segment(ctx: AcdcContext, next_: Callable[[], None]) -> None:
    """Open the 2D editor; block until the window closes."""
    ctx.images, _ = run_segment(ctx.images, ctx.segmentation)
    next_()


def run(ctx: AcdcContext) -> AcdcContext:
    """Run the segmentation viewer step and return ``ctx``."""
    ctx.images, _ = run_segment(ctx.images, ctx.segmentation)
    return ctx


def from_arrays(
    images: Sequence[ImageData],
    segmentation: SegmentationResult,
) -> AcdcContext:
    """Build a context from pre-loaded data."""
    return AcdcContext(images=coalesce_images(images), segmentation=segmentation)
