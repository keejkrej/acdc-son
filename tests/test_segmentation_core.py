"""Unit tests for manual segmentation core (no GUI)."""

from pathlib import Path

import numpy as np

from acdc.core.data import AcdcData, AcdcResult
from acdc.core import io, stack
from acdc.segment import editing
from acdc.segment.segment_model import SegmentationModel
from acdc.ui.lut import lut_with_hidden_labels


def test_apply_brush_and_save_roundtrip(tmp_path: Path) -> None:
    image = np.zeros((32, 48), dtype=np.uint16)
    image[10:20, 10:30] = 1000
    image_path = tmp_path / "cell.tif"
    import tifffile

    tifffile.imwrite(image_path, image)

    model = SegmentationModel()
    imaged = AcdcData.from_path(image_path)
    result = AcdcResult.empty_like(imaged)
    model.open([imaged], result, mask_path=tmp_path / "cellsegm.npz")
    sl = model.current_mask_slice()
    editing.apply_brush(sl, 15, 15, radius=3, label=2)
    model.set_mask_slice(sl)

    out = tmp_path / "cellsegm.npz"
    model.save_mask(out)
    loaded = io.load_mask(out)
    assert loaded.dtype == np.uint32
    assert loaded.shape == image.shape
    assert loaded[15, 15] == 2


def test_infer_shape_3d_z() -> None:
    stack_shape = stack.infer_shape((8, 64, 64))
    assert stack_shape.has_z and not stack_shape.has_time
    assert stack_shape.size_z == 8


def test_apply_brush_stroke_interpolates() -> None:
    mask = np.zeros((32, 32), dtype=np.uint32)
    editing.apply_brush_stroke(mask, 0, 20, 0, 0, radius=2, label=3)
    assert mask[0, 0] == 3
    assert mask[0, 10] == 3
    assert mask[0, 20] == 3
    assert mask[10, 10] == 0


def test_fill_label_holes() -> None:
    mask = np.zeros((20, 20), dtype=np.uint32)
    mask[2:18, 2:18] = 1
    mask[5:15, 5:15] = 0
    assert editing.fill_label_holes(mask, 1)
    assert mask[10, 10] == 1


def test_apply_label_visibility() -> None:
    mask = np.zeros((8, 8), dtype=np.uint32)
    mask[0:4, 0:4] = 1
    mask[4:8, 4:8] = 2
    unchanged = editing.apply_label_visibility(mask, set())
    assert unchanged is mask
    hidden = editing.apply_label_visibility(mask, {2})
    assert hidden[2, 2] == 1
    assert hidden[6, 6] == 0


def test_lut_with_hidden_labels() -> None:
    lut = np.zeros((8, 4), dtype=np.uint8)
    lut[1] = (255, 0, 0, 128)
    lut[2] = (0, 255, 0, 128)
    out = lut_with_hidden_labels(lut, {2})
    assert out[1, 3] == 128
    assert tuple(out[2]) == (0, 0, 0, 0)


def test_find_outer_boundaries() -> None:
    mask = np.zeros((8, 8), dtype=np.uint32)
    mask[2:6, 2:6] = 1
    boundary = editing.find_outer_boundaries(mask)
    assert boundary[2, 3]
    assert not boundary[3, 3]
    assert boundary.sum() == 12


def test_labels_to_contour_rgba() -> None:
    mask = np.zeros((8, 8), dtype=np.uint32)
    mask[2:6, 2:6] = 1
    lut = np.zeros((8, 4), dtype=np.uint8)
    lut[1] = (255, 0, 0, 114)
    rgba = editing.labels_to_contour_rgba(mask, lut)
    assert rgba[2, 3, 3] == 255
    assert rgba[3, 3, 3] == 0


def test_unique_labels_in_rect() -> None:
    mask = np.zeros((10, 10), dtype=np.uint32)
    mask[1:4, 1:4] = 1
    mask[6:8, 6:8] = 2
    assert editing.unique_labels_in_rect(mask, 0, 0, 5, 5) == [1]
    assert editing.unique_labels_in_rect(mask, 0, 0, 9, 9) == [1, 2]
    assert editing.unique_labels_in_rect(mask, 2, 2, 4, 4) == []
    assert editing.unique_labels_in_rect(mask, 0, 0, 2, 2) == []
    assert editing.unique_labels_in_rect(mask, 8, 8, 8, 8) == []


def test_label_bounding_box() -> None:
    mask = np.zeros((10, 10), dtype=np.uint32)
    mask[2:5, 3:7] = 3
    assert editing.label_bounding_box(mask, 3) == (2, 3, 4, 6)
    assert editing.label_bounding_box(mask, 0) is None


def test_model_label_at_and_rect() -> None:
    mask = np.zeros((10, 10), dtype=np.uint32)
    mask[2:5, 2:5] = 4
    mask[7:9, 7:9] = 5
    stack_shape = stack.infer_shape(mask.shape)
    model = SegmentationModel()
    model.image = np.zeros((10, 10), dtype=np.uint8)
    model.mask = mask
    model.stack_shape = stack_shape
    assert model.label_at(3, 3) == 4
    assert model.label_at(0, 0) == 0
    assert model.labels_in_rect(0, 0, 5, 5) == [4]
    assert model.labels_in_rect(0, 0, 9, 9) == [4, 5]
    assert model.labels_in_rect(3, 3, 5, 5) == []


def test_end_stroke_fills_holes(tmp_path: Path) -> None:
    image = np.ones((20, 20), dtype=np.uint8)
    image_path = tmp_path / "a.npy"
    np.save(image_path, image)
    model = SegmentationModel()
    imaged = AcdcData.from_path(image_path)
    result = AcdcResult.empty_like(imaged)
    model.open([imaged], result, mask_path=tmp_path / "cellsegm.npz")
    sl = model.current_mask_slice()
    sl[2:18, 2:18] = 1
    sl[5:15, 5:15] = 0
    model.set_mask_slice(sl)
    model.tool = "brush"
    model.begin_stroke()
    model.paint(3, 3)
    model.end_stroke()
    assert model.current_mask_slice()[10, 10] == 1


def test_all_label_ids() -> None:
    mask = np.zeros((4, 16, 16), dtype=np.uint32)
    mask[0, 2:6, 2:6] = 1
    mask[3, 8:10, 8:10] = 2
    stack_shape = stack.infer_shape(mask.shape)
    model = SegmentationModel()
    model.image = np.zeros(mask.shape, dtype=np.uint8)
    model.mask = mask
    model.stack_shape = stack_shape
    assert model.all_label_ids() == [1, 2]
    assert model.current_label_ids() == [1]
    model.z_index = 3
    assert model.current_label_ids() == [2]


def test_current_label_ids() -> None:
    mask = np.zeros((16, 16), dtype=np.uint32)
    mask[2:6, 2:6] = 1
    mask[8:10, 8:10] = 3
    stack_shape = stack.infer_shape(mask.shape)
    model = SegmentationModel()
    model.image = np.zeros((16, 16), dtype=np.uint8)
    model.mask = mask
    model.stack_shape = stack_shape
    assert model.current_label_ids() == [1, 3]


def test_undo(tmp_path: Path) -> None:
    image = np.ones((16, 16), dtype=np.uint8)
    image_path = tmp_path / "a.npy"
    np.save(image_path, image)
    model = SegmentationModel()
    imaged = AcdcData.from_path(image_path)
    result = AcdcResult.empty_like(imaged)
    model.open([imaged], result, mask_path=tmp_path / "asegm.npz")
    model.begin_stroke()
    model.paint(5, 5)
    model.end_stroke()
    assert model.current_mask_slice()[5, 5] != 0
    assert model.undo()
    assert model.current_mask_slice()[5, 5] == 0
