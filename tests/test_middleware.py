"""Tests for the middleware pipeline API."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from qtpy.QtCore import QTimer

from acdc.data import ImageData, SegmentationResult
from acdc.middleware import (
    AcdcContext,
    from_arrays,
    load,
    run_segment,
    segment,
    use,
    volume,
)


def test_from_arrays() -> None:
    imaged = ImageData.from_arrays(np.zeros((4, 4), dtype=np.uint8))
    result = SegmentationResult.empty_like(imaged)
    ctx = from_arrays([imaged], result)
    assert ctx.images == (imaged,)
    assert ctx.segmentation is result


def test_use_runs_in_order() -> None:
    order: list[str] = []
    ctx = from_arrays(
        [ImageData.from_arrays(np.zeros((4, 4), dtype=np.uint8))],
        SegmentationResult.empty_like(ImageData.from_arrays(np.zeros((4, 4), dtype=np.uint8))),
    )

    def first(c: AcdcContext, next_) -> None:
        order.append("first")
        next_()

    def second(c: AcdcContext, next_) -> None:
        order.append("second")
        next_()

    out = use(first, second)(ctx)
    assert order == ["first", "second"]
    assert out is ctx


def test_load_returns_context(tmp_path: Path) -> None:
    images_dir = tmp_path / "Position_1" / "Images"
    images_dir.mkdir(parents=True)
    import tifffile

    tifffile.imwrite(images_dir / "demo_s01_phase.tif", np.zeros((16, 16), dtype=np.uint16))
    (images_dir / "demo_s01_metadata.csv").write_text(
        "Description,values\nbasename,demo_s01_\nSizeT,1\nSizeZ,1\n"
        "channel_0_name,phase\n",
        encoding="utf-8",
    )
    ctx = load(tmp_path / "Position_1", channel="phase")
    assert isinstance(ctx, AcdcContext)
    assert len(ctx.images) == 1
    assert ctx.segmentation.mask.shape == ctx.images[0].image.shape


def test_segment_middleware_blocks_until_close() -> None:
    imaged = ImageData.from_arrays(np.zeros((6, 6), dtype=np.uint8))
    ctx = from_arrays([imaged], SegmentationResult.empty_like(imaged))

    def close_window() -> None:
        from acdc.segment.viewer import current_viewer

        viewer = current_viewer()
        assert viewer is not None
        viewer.view.close()

    QTimer.singleShot(0, close_window)
    use(segment)(ctx)
    assert ctx.images == (imaged,)


def test_run_segment_step() -> None:
    imaged = ImageData.from_arrays(np.zeros((6, 6), dtype=np.uint8))
    ctx = from_arrays([imaged], SegmentationResult.empty_like(imaged))

    def close_window() -> None:
        from acdc.segment.viewer import current_viewer

        viewer = current_viewer()
        assert viewer is not None
        viewer.view.close()

    QTimer.singleShot(0, close_window)
    run_segment(ctx)
    assert ctx.images == (imaged,)


def test_volume_middleware_blocks_until_close() -> None:
    imaged = ImageData.from_arrays(np.zeros((4, 8, 8), dtype=np.uint16))
    ctx = from_arrays([imaged], SegmentationResult.empty_like(imaged))

    def close_window() -> None:
        from acdc.volume.viewer import current_volume_viewer

        viewer = current_volume_viewer()
        assert viewer is not None
        viewer.view.close()

    QTimer.singleShot(0, close_window)
    use(volume)(ctx)
