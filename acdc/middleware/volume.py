"""3D volume viewer middleware."""

from __future__ import annotations

from collections.abc import Callable

from acdc.middleware.context import AcdcContext
from acdc.volume.viewer import run_volume


def volume(ctx: AcdcContext, next_: Callable[[], None]) -> None:
    """Open the 3D viewer; block until the window closes."""
    ctx.images, _ = run_volume(
        ctx.images, ctx.segmentation, t_index=ctx.t_index
    )
    next_()


def run(ctx: AcdcContext) -> AcdcContext:
    """Run the volume viewer step and return ``ctx``."""
    ctx.images, _ = run_volume(
        ctx.images, ctx.segmentation, t_index=ctx.t_index
    )
    return ctx
