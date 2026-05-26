"""Tests for 3D volume preparation and colormap helpers."""

from __future__ import annotations

import numpy as np
import pytest

from cellacdc.data import Experiment, ExperimentData, ImagedData
from cellacdc.segmentation import tools
from cellacdc.volume.prepare import (
    label_volume_for_vispy,
    mask_volume_zyx,
    normalize_image_volume,
    volume_zyx,
    voxel_display_scale,
)


def test_imaged_data_aliases() -> None:
    assert Experiment is ImagedData
    assert ExperimentData is ImagedData


def test_volume_zyx_from_zstack() -> None:
    image = np.zeros((8, 16, 16), dtype=np.uint8)
    image[3, 8, 8] = 100
    imaged = ImagedData.from_arrays(image)
    vol = volume_zyx(imaged)
    assert vol.shape == (8, 16, 16)
    assert vol[3, 8, 8] == 100


def test_mask_volume_zyx_matches_image_layout() -> None:
    image = np.zeros((4, 10, 10), dtype=np.uint8)
    mask = np.zeros((4, 10, 10), dtype=np.uint32)
    mask[1, 5, 5] = 2
    layout = tools.infer_layout(image.shape)
    imaged = ImagedData(image=image, layout=layout)
    from cellacdc.data import SegmentationResult

    result = SegmentationResult(mask)
    lab = mask_volume_zyx(result, layout)
    assert lab[1, 5, 5] == 2


def test_normalize_image_volume_scales_to_unit_interval() -> None:
    vol = np.array([[[0, 50], [100, 200]]], dtype=np.uint16)
    scaled, clim = normalize_image_volume(vol)
    assert clim == (0.0, 1.0)
    assert scaled.min() == 0.0
    assert scaled.max() == 1.0


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
