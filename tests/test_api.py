"""Tests for the public programmatic API (no GUI)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from acdc.data import (
    ImageData,
    SegmentationResult,
    default_segmentation,
)
from acdc.segment import io, tools
from acdc.segment.model import SegmentationModel


def test_imaged_from_arrays() -> None:
    image = np.zeros((16, 16), dtype=np.uint8)
    imaged = ImageData.from_arrays(image, title="demo")
    assert imaged.image is image
    assert imaged.layout.size_y == 16
    assert imaged.title == "demo"


def test_segmentation_result_empty_and_save(tmp_path: Path) -> None:
    imaged = ImageData.from_arrays(np.zeros((8, 8), dtype=np.uint8))
    result = SegmentationResult.empty_like(imaged)
    assert result.mask.shape == (8, 8)
    assert not result.dirty
    result.mask[3, 3] = 2
    result.dirty = True
    out = tmp_path / "mask.npz"
    result.save(out)
    assert not result.dirty
    loaded = io.load_mask(out)
    assert loaded[3, 3] == 2


def test_default_segmentation_loads_existing_mask(tmp_path: Path) -> None:
    image = np.zeros((8, 8), dtype=np.uint8)
    mask_path = tmp_path / "cellsegm.npz"
    mask = np.zeros((8, 8), dtype=np.uint32)
    mask[1, 1] = 5
    io.save_mask(mask_path, mask)
    imaged = ImageData.from_arrays(image, title="x")
    imaged = ImageData(
        image=imaged.image,
        layout=imaged.layout,
        mask_path=mask_path,
    )
    result = default_segmentation(imaged)
    assert result.mask[1, 1] == 5


def test_model_open_edits_result_in_place() -> None:
    image = np.ones((20, 20), dtype=np.uint8)
    imaged = ImageData.from_arrays(image)
    result = SegmentationResult.empty_like(imaged)
    model = SegmentationModel()
    model.open([imaged], result)
    assert model.has_data
    assert model.mask is result.mask
    model.tool = "brush"
    model.begin_stroke()
    model.paint(5, 5)
    model.end_stroke()
    assert result.mask[5, 5] == model.label_id
    assert result.dirty


def test_model_save_delegates_to_result(tmp_path: Path) -> None:
    imaged = ImageData.from_arrays(np.zeros((4, 4), dtype=np.uint8))
    result = SegmentationResult.empty_like(imaged)
    model = SegmentationModel()
    model.open([imaged], result)
    result.mask[0, 0] = 7
    dest = tmp_path / "out.npz"
    model.save_mask(dest)
    assert io.load_mask(dest)[0, 0] == 7
    assert not result.dirty


def test_imaged_from_path_single_position(tmp_path: Path) -> None:
    images = tmp_path / "Position_1" / "Images"
    images.mkdir(parents=True)
    import tifffile

    tifffile.imwrite(images / "demo_s01_phase.tif", np.zeros((16, 16), dtype=np.uint16))
    (images / "demo_s01_metadata.csv").write_text(
        "Description,values\nbasename,demo_s01_\nSizeT,1\nSizeZ,1\n"
        "channel_0_name,phase\n",
        encoding="utf-8",
    )
    imaged = ImageData.from_path(tmp_path / "Position_1", channel="phase")
    assert imaged.image_path == images / "demo_s01_phase.tif"
    assert imaged.mask_path == images / "demo_s01_segm.npz"
    assert imaged.layout.size_t == 1


def test_apply_brush_stroke_on_bound_result() -> None:
    imaged = ImageData.from_arrays(np.zeros((12, 12), dtype=np.uint8))
    result = SegmentationResult.empty_like(imaged)
    sl = tools.extract_slice(result.mask, imaged.layout, 0, 0)
    tools.apply_brush(sl, 6, 6, radius=2, label=3)
    assert result.mask[6, 6] == 3


def test_segmentation_viewer_open_binds_result() -> None:
    from acdc.segment.viewer import SegmentationViewer

    imaged = ImageData.from_arrays(np.zeros((6, 6), dtype=np.uint8))
    result = SegmentationResult.empty_like(imaged)
    viewer = SegmentationViewer()
    opened = viewer.open([imaged], result)
    assert opened is result
    assert viewer.model.mask is result.mask
    assert viewer.result is result


def test_run_segment_returns_images_and_segmentation() -> None:
    from qtpy.QtCore import QTimer

    from acdc.segment.viewer import current_viewer, run_segment

    imaged = ImageData.from_arrays(np.zeros((6, 6), dtype=np.uint8))
    result = SegmentationResult.empty_like(imaged)

    def close_window() -> None:
        viewer = current_viewer()
        assert viewer is not None
        viewer.view.close()

    QTimer.singleShot(0, close_window)
    out_images, opened = run_segment([imaged], result)
    assert opened is result
    assert out_images == (imaged,)


def test_run_volume_returns_images_and_segmentation() -> None:
    from qtpy.QtCore import QTimer

    from acdc.volume.viewer import current_volume_viewer, run_volume

    image = np.zeros((4, 8, 8), dtype=np.uint16)
    imaged = ImageData.from_arrays(image)
    result = SegmentationResult.empty_like(imaged)

    def close_window() -> None:
        viewer = current_volume_viewer()
        assert viewer is not None
        viewer.view.close()

    QTimer.singleShot(0, close_window)
    out_images, opened = run_volume([imaged], result)
    assert opened is result
    assert out_images == (imaged,)


def test_volume_accepts_channel_list() -> None:
    from acdc.volume.viewer import VolumeViewer

    primary = ImageData.from_arrays(np.zeros((4, 8, 8), dtype=np.uint16))
    overlay = ImageData.from_arrays(np.ones((4, 8, 8), dtype=np.uint16) * 100, title="gfp")
    result = SegmentationResult.empty_like(primary)
    viewer = VolumeViewer()
    viewer.open([primary, overlay], result)
    assert viewer.model.has_data
    assert viewer.model.channels == [primary, overlay]
    assert viewer.result is result


def test_load_returns_images_and_segmentation(tmp_path: Path) -> None:
    from acdc.middleware import load

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
    assert len(ctx.images) == 1
    assert ctx.segmentation.mask.shape == ctx.images[0].image.shape


def test_volume_viewer_open_without_show() -> None:
    from acdc.volume.viewer import VolumeViewer

    image = np.zeros((4, 8, 8), dtype=np.uint16)
    image[:, 3:5, 3:5] = 500
    imaged = ImageData.from_arrays(image)
    result = SegmentationResult.empty_like(imaged)
    result.mask[:, 3:5, 3:5] = 1
    viewer = VolumeViewer()
    opened = viewer.open([imaged], result)
    assert opened is result
    assert viewer.primary is imaged
    assert viewer.model.has_data
    assert viewer.view.get_hidden_label_ids() == set()
