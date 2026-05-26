"""Unit tests for manual segmentation core (no GUI)."""

from pathlib import Path

import numpy as np

from cellacdc.segmentation import io, tools
from cellacdc.segmentation.model import SegmentationModel


def test_apply_brush_and_save_roundtrip(tmp_path: Path) -> None:
    image = np.zeros((32, 48), dtype=np.uint16)
    image[10:20, 10:30] = 1000
    image_path = tmp_path / "cell.tif"
    import tifffile

    tifffile.imwrite(image_path, image)

    model = SegmentationModel()
    model.load_image(image_path)
    sl = model.current_mask_slice()
    tools.apply_brush(sl, 15, 15, radius=3, label=2)
    model.set_mask_slice(sl)

    out = tmp_path / "cellsegm.npz"
    model.save_mask(out)
    loaded = io.load_mask(out)
    assert loaded.dtype == np.uint32
    assert loaded.shape == image.shape
    assert loaded[15, 15] == 2


def test_infer_layout_3d_z() -> None:
    layout = tools.infer_layout((8, 64, 64))
    assert layout.has_z and not layout.has_time
    assert layout.size_z == 8


def test_apply_brush_stroke_interpolates() -> None:
    mask = np.zeros((32, 32), dtype=np.uint32)
    tools.apply_brush_stroke(mask, 0, 20, 0, 0, radius=2, label=3)
    assert mask[0, 0] == 3
    assert mask[0, 10] == 3
    assert mask[0, 20] == 3
    assert mask[10, 10] == 0


def test_fill_label_holes() -> None:
    mask = np.zeros((20, 20), dtype=np.uint32)
    mask[2:18, 2:18] = 1
    mask[5:15, 5:15] = 0
    assert tools.fill_label_holes(mask, 1)
    assert mask[10, 10] == 1


def test_filter_visible_labels() -> None:
    mask = np.zeros((8, 8), dtype=np.uint32)
    mask[0:4, 0:4] = 1
    mask[4:8, 4:8] = 2
    filtered = tools.filter_visible_labels(mask, {1})
    assert filtered[2, 2] == 1
    assert filtered[6, 6] == 0
    assert tools.filter_visible_labels(mask, set()).max() == 0


def test_apply_label_visibility() -> None:
    mask = np.zeros((8, 8), dtype=np.uint32)
    mask[0:4, 0:4] = 1
    mask[4:8, 4:8] = 2
    unchanged = tools.apply_label_visibility(mask, set())
    assert unchanged is mask
    hidden = tools.apply_label_visibility(mask, {2})
    assert hidden[2, 2] == 1
    assert hidden[6, 6] == 0


def test_end_stroke_fills_holes(tmp_path: Path) -> None:
    image = np.ones((20, 20), dtype=np.uint8)
    image_path = tmp_path / "a.npy"
    np.save(image_path, image)
    model = SegmentationModel()
    model.load_image(image_path)
    sl = model.current_mask_slice()
    sl[2:18, 2:18] = 1
    sl[5:15, 5:15] = 0
    model.set_mask_slice(sl)
    model.begin_stroke()
    model.paint(3, 3)
    model.end_stroke()
    assert model.current_mask_slice()[10, 10] == 1


def test_all_label_ids() -> None:
    mask = np.zeros((4, 16, 16), dtype=np.uint32)
    mask[0, 2:6, 2:6] = 1
    mask[3, 8:10, 8:10] = 2
    layout = tools.infer_layout(mask.shape)
    model = SegmentationModel()
    model.image = np.zeros(mask.shape, dtype=np.uint8)
    model.mask = mask
    model.layout = layout
    assert model.all_label_ids() == [1, 2]
    assert model.current_label_ids() == [1]
    model.z_index = 3
    assert model.current_label_ids() == [2]


def test_current_label_ids() -> None:
    mask = np.zeros((16, 16), dtype=np.uint32)
    mask[2:6, 2:6] = 1
    mask[8:10, 8:10] = 3
    layout = tools.infer_layout(mask.shape)
    model = SegmentationModel()
    model.image = np.zeros((16, 16), dtype=np.uint8)
    model.mask = mask
    model.layout = layout
    assert model.current_label_ids() == [1, 3]


def test_undo(tmp_path: Path) -> None:
    image = np.ones((16, 16), dtype=np.uint8)
    image_path = tmp_path / "a.npy"
    np.save(image_path, image)
    model = SegmentationModel()
    model.load_image(image_path)
    model.begin_stroke()
    model.paint(5, 5)
    model.end_stroke()
    assert model.current_mask_slice()[5, 5] != 0
    assert model.undo()
    assert model.current_mask_slice()[5, 5] == 0
