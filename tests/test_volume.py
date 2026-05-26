"""Tests for 3D volume preparation and colormap helpers."""

from __future__ import annotations

import numpy as np
import pytest

from acdc.utils.blend import display_opacities
from acdc.core.data import AcdcData
from acdc.core import stack
from acdc.volume.prepare import (
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
    imaged = AcdcData.from_arrays(image)
    vol = volume_zyx(imaged)
    assert vol.shape == (8, 16, 16)
    assert vol[3, 8, 8] == 100


def test_mask_volume_zyx_matches_image_layout() -> None:
    image = np.zeros((4, 10, 10), dtype=np.uint8)
    mask = np.zeros((4, 10, 10), dtype=np.uint32)
    mask[1, 5, 5] = 2
    stack_shape = stack.infer_shape(image.shape)
    imaged = AcdcData(image=image, stack_shape=stack_shape)
    from acdc.core.data import AcdcResult

    result = AcdcResult(mask)
    lab = mask_volume_zyx(result, stack_shape)
    assert lab[1, 5, 5] == 2


def test_normalize_image_volume_scales_to_unit_interval() -> None:
    vol = np.array([[[0, 50], [100, 200]]], dtype=np.uint16)
    scaled, clim = normalize_image_volume(vol)
    assert scaled.min() >= 0.0
    assert scaled.max() <= 1.0
    assert clim[1] > clim[0]


def test_normalize_image_stack_volume_uses_full_stack_levels() -> None:
    volume = np.zeros((8, 8, 8), dtype=np.uint16)
    volume[2, 1:7, 1:7] = 100
    volume[4, 4, 4] = 60000
    stack_shape = stack.infer_shape(volume.shape)
    z_slice = volume[2]
    scaled, clim = normalize_image_stack_volume(z_slice, volume, stack_shape)
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


def test_volume_canvas_dual_channels_use_napari_style_gl_blend() -> None:
    pytest.importorskip("vispy")
    import os

    os.environ.setdefault("QT_API", "pyside6")
    import vispy

    vispy.use(app="pyside6")
    from qtpy.QtWidgets import QApplication

    from acdc.volume.canvas import VolumeCanvas
    from acdc.volume.gl_blend import volume_gl_state

    app = QApplication.instance() or QApplication([])
    canvas = VolumeCanvas()
    primary = np.zeros((8, 8, 8), dtype=np.float32)
    primary[2, 2, 2] = 1.0
    secondary = np.zeros((8, 8, 8), dtype=np.float32)
    secondary[5, 5, 5] = 1.0
    canvas.set_image_channels(
        [primary, secondary],
        clims=[(0.0, 1.0), (0.0, 1.0)],
        label_volume=None,
        label_lut_size=2,
    )

    assert len(canvas._image_channels) == 2
    ch0, ch1 = canvas._image_channels
    assert ch0.node.method == "mip"
    assert ch1.node.method == "mip"
    assert ch0.node.order < ch1.node.order
    image_blend = ch0.node._vshare.gl_state["blend_func"]
    secondary_blend = ch1.node._vshare.gl_state["blend_func"]
    assert image_blend[:2] == volume_gl_state(
        "translucent_no_depth", first_visible=True
    )["blend_func"][:2]
    assert secondary_blend[:2] == volume_gl_state("additive", first_visible=False)[
        "blend_func"
    ][:2]


def test_volume_canvas_single_active_channel_uses_first_visible_additive() -> None:
    pytest.importorskip("vispy")
    import os

    os.environ.setdefault("QT_API", "pyside6")
    import vispy

    vispy.use(app="pyside6")
    from qtpy.QtWidgets import QApplication

    from acdc.volume.canvas import VolumeCanvas
    from acdc.volume.gl_blend import volume_gl_state

    app = QApplication.instance() or QApplication([])
    canvas = VolumeCanvas()
    canvas.set_image_channels(
        [np.zeros((4, 4, 4), dtype=np.float32), np.ones((4, 4, 4), dtype=np.float32) * 0.5],
        clims=[(0.0, 1.0), (0.0, 1.0)],
        label_volume=None,
        label_lut_size=2,
    )
    canvas.set_channel_weights([0.0, 1.0])

    ch1 = canvas._image_channels[1]
    secondary_blend = ch1.node._vshare.gl_state["blend_func"]
    assert secondary_blend[:2] == volume_gl_state(
        "translucent_no_depth", first_visible=True
    )["blend_func"][:2]


def test_volume_canvas_channel_weights_use_normalized_opacity() -> None:
    pytest.importorskip("vispy")
    import os

    os.environ.setdefault("QT_API", "pyside6")
    import vispy

    vispy.use(app="pyside6")
    from qtpy.QtWidgets import QApplication

    from acdc.volume.canvas import VolumeCanvas

    app = QApplication.instance() or QApplication([])
    canvas = VolumeCanvas()
    primary = np.zeros((8, 8, 8), dtype=np.float32)
    primary[2, 2, 2] = 1.0
    secondary = np.zeros((8, 8, 8), dtype=np.float32)
    secondary[5, 5, 5] = 1.0
    canvas.set_image_channels(
        [primary, secondary],
        clims=[(0.0, 1.0), (0.0, 1.0)],
        label_volume=None,
        label_lut_size=2,
    )
    canvas.set_channel_weights([0.0, 1.0])
    canvas.set_image_seg_blend(0)

    opacities, _seg = display_opacities([0.0, 1.0], 0)
    assert canvas._image_channels[0].node.opacity == pytest.approx(opacities[0])
    assert canvas._image_channels[1].node.opacity == pytest.approx(opacities[1])
    assert canvas._image_channels[0].node._last_data[2, 2, 2] == 1.0
    assert canvas._image_channels[1].node._last_data[5, 5, 5] == 1.0


def test_volume_canvas_uses_mip_for_image_channels() -> None:
    pytest.importorskip("vispy")
    import os

    os.environ.setdefault("QT_API", "pyside6")
    import vispy

    vispy.use(app="pyside6")
    from qtpy.QtWidgets import QApplication

    from acdc.volume.canvas import VolumeCanvas

    app = QApplication.instance() or QApplication([])
    canvas = VolumeCanvas()
    canvas._ensure_vispy()
    canvas.set_image_channels(
        [np.zeros((2, 2, 2), dtype=np.float32)],
        clims=[(0.0, 1.0)],
        label_volume=None,
        label_lut_size=2,
    )
    assert canvas._image_channels[0].node.method == "mip"
    assert canvas._label_node.method == "translucent"


def test_pg_colormap_to_vispy() -> None:
    pytest.importorskip("vispy")
    import pyqtgraph as pg

    from acdc.volume.cmaps import label_lut_to_vispy, pg_colormap_to_vispy

    lut_widget = pg.HistogramLUTWidget()
    lut_widget.item.gradient.loadPreset("grey")
    pg_cmap = lut_widget.item.gradient.colorMap()
    vispy_cmap = pg_colormap_to_vispy(pg_cmap, n=16)
    assert len(np.asarray(vispy_cmap.colors)) == 16

    lut = np.zeros((8, 4), dtype=np.uint8)
    lut[1] = (255, 0, 0, 128)
    label_cmap = label_lut_to_vispy(lut)
    assert label_cmap.interpolation == "zero"
