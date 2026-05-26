"""Tests for the public programmatic API (no GUI)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cellacdc.data import (
    Experiment,
    ExperimentData,
    SegmentationResult,
    default_segmentation,
)
from cellacdc.segmentation import io, tools
from cellacdc.segmentation.model import SegmentationModel


def test_experiment_from_arrays() -> None:
    image = np.zeros((16, 16), dtype=np.uint8)
    exp = Experiment.from_arrays(image, title="demo")
    assert exp.image is image
    assert exp.layout.size_y == 16
    assert exp.title == "demo"


def test_experiment_data_alias() -> None:
    assert ExperimentData is Experiment


def test_segmentation_result_empty_and_save(tmp_path: Path) -> None:
    exp = Experiment.from_arrays(np.zeros((8, 8), dtype=np.uint8))
    result = SegmentationResult.empty_like(exp)
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
    exp = Experiment.from_arrays(image, title="x")
    exp = Experiment(
        image=exp.image,
        layout=exp.layout,
        mask_path=mask_path,
    )
    result = default_segmentation(exp)
    assert result.mask[1, 1] == 5


def test_model_open_edits_result_in_place() -> None:
    image = np.ones((20, 20), dtype=np.uint8)
    exp = Experiment.from_arrays(image)
    result = SegmentationResult.empty_like(exp)
    model = SegmentationModel()
    model.open(exp, result)
    assert model.has_data
    assert model.mask is result.mask
    model.tool = "brush"
    model.begin_stroke()
    model.paint(5, 5)
    model.end_stroke()
    assert result.mask[5, 5] == model.label_id
    assert result.dirty


def test_model_save_delegates_to_result(tmp_path: Path) -> None:
    exp = Experiment.from_arrays(np.zeros((4, 4), dtype=np.uint8))
    result = SegmentationResult.empty_like(exp)
    model = SegmentationModel()
    model.open(exp, result)
    result.mask[0, 0] = 7
    dest = tmp_path / "out.npz"
    model.save_mask(dest)
    assert io.load_mask(dest)[0, 0] == 7
    assert not result.dirty


def test_experiment_from_path_single_position(tmp_path: Path) -> None:
    images = tmp_path / "Position_1" / "Images"
    images.mkdir(parents=True)
    import tifffile

    tifffile.imwrite(images / "demo_s01_phase.tif", np.zeros((16, 16), dtype=np.uint16))
    (images / "demo_s01_metadata.csv").write_text(
        "Description,values\nbasename,demo_s01_\nSizeT,1\nSizeZ,1\n"
        "channel_0_name,phase\n",
        encoding="utf-8",
    )
    exp = Experiment.from_path(tmp_path / "Position_1", channel="phase")
    assert exp.image_path == images / "demo_s01_phase.tif"
    assert exp.mask_path == images / "demo_s01_segm.npz"
    assert exp.layout.size_t == 1


def test_apply_brush_stroke_on_bound_result() -> None:
    exp = Experiment.from_arrays(np.zeros((12, 12), dtype=np.uint8))
    result = SegmentationResult.empty_like(exp)
    sl = tools.extract_slice(result.mask, exp.layout, 0, 0)
    tools.apply_brush(sl, 6, 6, radius=2, label=3)
    assert result.mask[6, 6] == 3


def test_segmentation_viewer_open_binds_result() -> None:
    from cellacdc.viewer import SegmentationViewer

    exp = Experiment.from_arrays(np.zeros((6, 6), dtype=np.uint8))
    result = SegmentationResult.empty_like(exp)
    viewer = SegmentationViewer()
    opened = viewer.open(exp, result=result)
    assert opened is result
    assert viewer.model.mask is result.mask
    assert viewer.result is result


def test_imshow_returns_viewer_and_result() -> None:
    from cellacdc.viewer import imshow

    exp = Experiment.from_arrays(np.zeros((6, 6), dtype=np.uint8))
    result = SegmentationResult.empty_like(exp)
    viewer, opened = imshow(exp, result=result, show=False)
    assert viewer.model.has_data
    assert opened is result
    assert opened.mask is result.mask
