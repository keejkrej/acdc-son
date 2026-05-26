"""Tests for the public programmatic API (no GUI)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from acdc.core.data import (
    AcdcData,
    AcdcResult,
    load,
    load_segmentation,
    save_segmentation,
)
from acdc.core import io
from acdc.core import stack
from acdc.segment.model import SegmentationModel


def test_imaged_from_arrays() -> None:
    image = np.zeros((16, 16), dtype=np.uint8)
    imaged = AcdcData.from_arrays(image, name="demo")
    assert imaged.image is image
    assert imaged.stack_shape.size_y == 16
    assert imaged.name == "demo"


def test_segmentation_result_empty_and_save(tmp_path: Path) -> None:
    imaged = AcdcData.from_arrays(np.zeros((8, 8), dtype=np.uint8))
    result = AcdcResult.empty_like(imaged)
    assert result.mask.shape == (8, 8)
    result.mask[3, 3] = 2
    out = tmp_path / "mask.npz"
    save_segmentation(result, out)
    loaded = io.load_mask(out)
    assert loaded[3, 3] == 2


def test_load_segmentation_reads_existing_mask(tmp_path: Path) -> None:
    image = np.zeros((8, 8), dtype=np.uint8)
    image_path = tmp_path / "cell.tif"
    mask_path = tmp_path / "cellsegm.npz"
    import tifffile

    tifffile.imwrite(image_path, image)
    mask = np.zeros((8, 8), dtype=np.uint32)
    mask[1, 1] = 5
    io.save_mask(mask_path, mask)
    imaged = AcdcData.from_path(image_path)
    result = load_segmentation(mask_path, like=imaged)
    assert result.mask[1, 1] == 5


def test_model_open_edits_result_in_place() -> None:
    image = np.ones((20, 20), dtype=np.uint8)
    imaged = AcdcData.from_arrays(image)
    result = AcdcResult.empty_like(imaged)
    model = SegmentationModel()
    model.open([imaged], result)
    assert model.has_data
    assert model.mask is result.mask
    model.tool = "brush"
    model.begin_stroke()
    model.paint(5, 5)
    model.end_stroke()
    assert result.mask[5, 5] == model.label_id
    assert not model.saved


def test_model_save_delegates_to_result(tmp_path: Path) -> None:
    imaged = AcdcData.from_arrays(np.zeros((4, 4), dtype=np.uint8))
    result = AcdcResult.empty_like(imaged)
    model = SegmentationModel()
    model.open([imaged], result)
    result.mask[0, 0] = 7
    dest = tmp_path / "out.npz"
    model.save_mask(dest)
    assert io.load_mask(dest)[0, 0] == 7
    assert model.saved


def test_imaged_from_experiment_single_position(tmp_path: Path) -> None:
    images = tmp_path / "Position_1" / "Images"
    images.mkdir(parents=True)
    import tifffile

    tifffile.imwrite(images / "demo_s01_phase.tif", np.zeros((16, 16), dtype=np.uint16))
    (images / "demo_s01_metadata.csv").write_text(
        "Description,values\nbasename,demo_s01_\nSizeT,1\nSizeZ,1\n"
        "channel_0_name,phase\n",
        encoding="utf-8",
    )
    loaded = AcdcData.from_experiment(tmp_path / "Position_1", channel="phase")
    imaged = loaded[0]
    assert imaged.name == "Position_1 / phase"
    assert imaged.stack_shape.size_t == 1
    images, result = load(tmp_path / "Position_1", channel="phase")
    assert images[0].name == imaged.name
    assert images[0].image.shape == imaged.image.shape
    assert result.mask.shape == imaged.image.shape


def test_apply_brush_stroke_on_bound_result() -> None:
    imaged = AcdcData.from_arrays(np.zeros((12, 12), dtype=np.uint8))
    result = AcdcResult.empty_like(imaged)
    sl = stack.extract_slice(result.mask, imaged.stack_shape, 0, 0)
    from acdc.segment import editing

    editing.apply_brush(sl, 6, 6, radius=2, label=3)
    assert result.mask[6, 6] == 3


def test_segmentation_viewer_open_binds_result() -> None:
    from acdc.segment.viewer import SegmentationViewer

    imaged = AcdcData.from_arrays(np.zeros((6, 6), dtype=np.uint8))
    result = AcdcResult.empty_like(imaged)
    viewer = SegmentationViewer()
    opened = viewer.open([imaged], result)
    assert opened is result
    assert viewer.model.mask is result.mask
    assert viewer.result is result


def test_run_segment_returns_images_and_segmentation() -> None:
    from qtpy.QtCore import QTimer

    from acdc.segment.viewer import current_viewer, run_segment

    imaged = AcdcData.from_arrays(np.zeros((6, 6), dtype=np.uint8))
    result = AcdcResult.empty_like(imaged)

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
    imaged = AcdcData.from_arrays(image)
    result = AcdcResult.empty_like(imaged)

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

    primary = AcdcData.from_arrays(np.zeros((4, 8, 8), dtype=np.uint16))
    overlay = AcdcData.from_arrays(np.ones((4, 8, 8), dtype=np.uint16) * 100, name="gfp")
    result = AcdcResult.empty_like(primary)
    viewer = VolumeViewer()
    viewer.open([primary, overlay], result)
    assert viewer.model.has_data
    assert viewer.model.channels == [primary, overlay]
    assert viewer.result is result


def test_load_returns_images_and_segmentation(tmp_path: Path) -> None:
    from acdc.middleware import load as load_ctx

    images_dir = tmp_path / "Position_1" / "Images"
    images_dir.mkdir(parents=True)
    import tifffile

    tifffile.imwrite(images_dir / "demo_s01_phase.tif", np.zeros((16, 16), dtype=np.uint16))
    (images_dir / "demo_s01_metadata.csv").write_text(
        "Description,values\nbasename,demo_s01_\nSizeT,1\nSizeZ,1\n"
        "channel_0_name,phase\n",
        encoding="utf-8",
    )
    ctx = load_ctx(tmp_path / "Position_1", channel="phase")
    assert len(ctx.images) == 1
    assert ctx.segmentation.mask.shape == ctx.images[0].image.shape


def test_volume_viewer_open_without_show() -> None:
    from acdc.volume.viewer import VolumeViewer

    image = np.zeros((4, 8, 8), dtype=np.uint16)
    image[:, 3:5, 3:5] = 500
    imaged = AcdcData.from_arrays(image)
    result = AcdcResult.empty_like(imaged)
    result.mask[:, 3:5, 3:5] = 1
    viewer = VolumeViewer()
    opened = viewer.open([imaged], result)
    assert opened is result
    assert viewer.primary is imaged
    assert viewer.model.has_data
    assert viewer.view.get_hidden_label_ids() == set()
