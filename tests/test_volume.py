"""Tests for 3D volume preparation and colormap helpers."""

from __future__ import annotations

import numpy as np
import pytest

from cellacdc.data import ImageData
from cellacdc.segmentation import tools
from cellacdc.volume.prepare import (
    label_volume_for_vispy,
    mask_volume_zyx,
    normalize_image_stack_volume,
    normalize_image_volume,
    volume_zyx,
    voxel_display_scale,
)


def test_volume_zyx_from_zstack() -> None:
    image = np.zeros((8, 16, 16), dtype=np.uint8)
    image[3, 8, 8] = 100
    imaged = ImageData.from_arrays(image)
    vol = volume_zyx(imaged)
    assert vol.shape == (8, 16, 16)
    assert vol[3, 8, 8] == 100


def test_mask_volume_zyx_matches_image_layout() -> None:
    image = np.zeros((4, 10, 10), dtype=np.uint8)
    mask = np.zeros((4, 10, 10), dtype=np.uint32)
    mask[1, 5, 5] = 2
    layout = tools.infer_layout(image.shape)
    imaged = ImageData(image=image, layout=layout)
    from cellacdc.data import SegmentationResult

    result = SegmentationResult(mask)
    lab = mask_volume_zyx(result, layout)
    assert lab[1, 5, 5] == 2


def test_normalize_image_volume_scales_to_unit_interval() -> None:
    vol = np.array([[[0, 50], [100, 200]]], dtype=np.uint16)
    scaled, clim = normalize_image_volume(vol)
    assert scaled.min() >= 0.0
    assert scaled.max() <= 1.0
    assert clim[1] > clim[0]


def test_normalize_image_stack_volume_uses_full_stack_levels() -> None:
    stack = np.zeros((8, 8, 8), dtype=np.uint16)
    stack[2, 1:7, 1:7] = 100
    stack[4, 4, 4] = 60000
    layout = tools.infer_layout(stack.shape)
    vol = stack[2]
    scaled, clim = normalize_image_stack_volume(vol, stack, layout)
    assert scaled[2, 2] == pytest.approx(100 / 60000)
    assert clim[1] > clim[0]


def test_label_volume_for_vispy_lut_size() -> None:
    vol = np.zeros((4, 4, 4), dtype=np.uint32)
    vol[1, 1, 1] = 12
    lab, lut_size = label_volume_for_vispy(vol)
    assert lab.dtype == np.float32
    assert lut_size == 13


def test_label_volume_for_vispy_small_ids_use_tight_lut() -> None:
    vol = np.zeros((4, 4, 4), dtype=np.uint32)
    vol[0, 0, 0] = 1
    vol[0, 0, 1] = 4
    _lab, lut_size = label_volume_for_vispy(vol)
    assert lut_size == 5


def test_voxel_display_scale_matches_cell_acdc_ratio() -> None:
    assert voxel_display_scale(2.0, 1.0, 1.0) == (1.0, 1.0, 2.0)
    assert voxel_display_scale(1.0, 0.5, 0.5) == (1.0, 1.0, 2.0)


def test_volume_canvas_dual_volumes_use_opacity_blend() -> None:
    pytest.importorskip("vispy")
    import os

    os.environ.setdefault("QT_API", "pyside6")
    import vispy

    vispy.use(app="pyside6")
    from qtpy.QtWidgets import QApplication

    from cellacdc.volume.canvas import VolumeCanvas

    app = QApplication.instance() or QApplication([])
    canvas = VolumeCanvas()
    primary = np.zeros((8, 8, 8), dtype=np.float32)
    primary[2, 2, 2] = 1.0
    secondary = np.zeros((8, 8, 8), dtype=np.float32)
    secondary[5, 5, 5] = 1.0
    canvas.set_volumes(primary, None, label_lut_size=2, image_clim=(0.0, 1.0))
    canvas.set_secondary_volume(secondary, clim=(0.0, 1.0))
    canvas.set_image_seg_blend(0)
    canvas.set_primary_secondary_blend(100)

    assert canvas._secondary_node is not None
    assert canvas._secondary_node.visible is True
    assert canvas._image_node.opacity == 0.0
    assert canvas._secondary_node.opacity == 1.0
    assert canvas._image_node._last_data[2, 2, 2] == 1.0
    assert canvas._secondary_node._last_data[5, 5, 5] == 1.0


def test_volume_canvas_uses_mip_for_image_channels() -> None:
    pytest.importorskip("vispy")
    import os

    os.environ.setdefault("QT_API", "pyside6")
    import vispy

    vispy.use(app="pyside6")
    from qtpy.QtWidgets import QApplication

    from cellacdc.volume.canvas import VolumeCanvas

    app = QApplication.instance() or QApplication([])
    canvas = VolumeCanvas()
    canvas._ensure_vispy()
    assert canvas._image_node.method == "mip"
    assert canvas._label_node.method == "translucent"


def test_pg_colormap_to_vispy() -> None:
    pytest.importorskip("vispy")
    import pyqtgraph as pg

    from cellacdc.volume.cmaps import label_lut_to_vispy, pg_colormap_to_vispy

    lut_widget = pg.HistogramLUTWidget()
    lut_widget.item.gradient.loadPreset("grey")
    pg_cmap = lut_widget.item.gradient.colorMap()
    vispy_cmap = pg_colormap_to_vispy(pg_cmap, n=16)
    assert len(np.asarray(vispy_cmap.colors)) == 16

    lut = np.zeros((8, 4), dtype=np.uint8)
    lut[1] = (255, 0, 0, 128)
    label_cmap = label_lut_to_vispy(lut)
    assert label_cmap.interpolation == "zero"
