"""3D volume viewer middleware."""

from __future__ import annotations

from collections.abc import Callable

from acdc.middleware.context import AcdcContext
from acdc.volume.volume_viewer import run_volume as open_volume


def run_volume(ctx: AcdcContext, next_: Callable[[], None]) -> None:
    """Open the 3D viewer; block until the window closes."""
    ctx.images, _ = open_volume(
        ctx.images, ctx.segmentation, t_index=ctx.t_index
    )
    next_()
