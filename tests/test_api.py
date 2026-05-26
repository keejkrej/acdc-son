"""Tests for the public programmatic API (no GUI)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cellacdc.data import (
    Experiment,
    ExperimentData,
    ImagedData,
    SegmentationResult,
    default_segmentation,
)
from cellacdc.segmentation import io, tools
from cellacdc.segmentation.model import SegmentationModel


def test_imaged_from_arrays() -> None:
    image = np.zeros((16, 16), dtype=np.uint8)
    imaged = ImagedData.from_arrays(image, title="demo")
    assert imaged.image is image
    assert imaged.layout.size_y == 16
    assert imaged.title == "demo"


def test_experiment_data_alias() -> None:
    assert ExperimentData is ImagedData
    assert Experiment is ImagedData


def test_segmentation_result_empty_and_save(tmp_path: Path) -> None:
    imaged = ImagedData.from_arrays(np.zeros((8, 8), dtype=np.uint8))
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
    imaged = ImagedData.from_arrays(image, title="x")
    imaged = ImagedData(
        image=imaged.image,
        layout=imaged.layout,
        mask_path=mask_path,
    )
    result = default_segmentation(imaged)
    assert result.mask[1, 1] == 5


def test_model_open_edits_result_in_place() -> None:
    image = np.ones((20, 20), dtype=np.uint8)
    imaged = ImagedData.from_arrays(image)
    result = SegmentationResult.empty_like(imaged)
    model = SegmentationModel()
    model.open(imaged, result)
    assert model.has_data
    assert model.mask is result.mask
    model.tool = "brush"
    model.begin_stroke()
    model.paint(5, 5)
    model.end_stroke()
    assert result.mask[5, 5] == model.label_id
    assert result.dirty


def test_model_save_delegates_to_result(tmp_path: Path) -> None:
    imaged = ImagedData.from_arrays(np.zeros((4, 4), dtype=np.uint8))
    result = SegmentationResult.empty_like(imaged)
    model = SegmentationModel()
    model.open(imaged, result)
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
    imaged = ImagedData.from_path(tmp_path / "Position_1", channel="phase")
    assert imaged.image_path == images / "demo_s01_phase.tif"
    assert imaged.mask_path == images / "demo_s01_segm.npz"
    assert imaged.layout.size_t == 1


def test_apply_brush_stroke_on_bound_result() -> None:
    imaged = ImagedData.from_arrays(np.zeros((12, 12), dtype=np.uint8))
    result = SegmentationResult.empty_like(imaged)
    sl = tools.extract_slice(result.mask, imaged.layout, 0, 0)
    tools.apply_brush(sl, 6, 6, radius=2, label=3)
    assert result.mask[6, 6] == 3


def test_segmentation_viewer_open_binds_result() -> None:
    from cellacdc.viewer import SegmentationViewer

    imaged = ImagedData.from_arrays(np.zeros((6, 6), dtype=np.uint8))
    result = SegmentationResult.empty_like(imaged)
    viewer = SegmentationViewer()
    opened = viewer.open(imaged, result=result)
    assert opened is result
    assert viewer.model.mask is result.mask
    assert viewer.result is result


def test_imshow_returns_viewer_and_result() -> None:
    from cellacdc.viewer import imshow

    imaged = ImagedData.from_arrays(np.zeros((6, 6), dtype=np.uint8))
    result = SegmentationResult.empty_like(imaged)
    viewer, opened = imshow(imaged, result=result, show=False)
    assert viewer.model.has_data
    assert opened is result
    assert opened.mask is result.mask


def test_volume_viewer_open_without_show() -> None:
    from cellacdc.volume import VolumeViewer

    image = np.zeros((4, 8, 8), dtype=np.uint16)
    image[:, 3:5, 3:5] = 500
    imaged = ImagedData.from_arrays(image)
    result = SegmentationResult.empty_like(imaged)
    result.mask[:, 3:5, 3:5] = 1
    viewer = VolumeViewer()
    opened = viewer.open(imaged, result=result)
    assert opened is result
    assert viewer.imaged is imaged
    assert viewer.model.has_data
    assert viewer.window.get_hidden_label_ids() == set()
