"""Vispy volume canvas with dual LUT bars."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import QHBoxLayout, QSizePolicy, QVBoxLayout, QWidget

from cellacdc.volume.cmaps import label_lut_to_vispy, pg_colormap_to_vispy
from cellacdc.volume.lut import VolumeImageLutBar, VolumeLabelLutBar
from cellacdc.volume.prepare import voxel_display_scale


class _LutColumn(QWidget):
    """Single LUT bar column; stretches to match the 3D canvas height."""

    def __init__(self, lut_item: pg.HistogramLUTItem, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self._glw = pg.GraphicsLayoutWidget()
        layout.addWidget(self._glw, stretch=1)
        self._glw.addItem(lut_item, row=0, col=0)
        self.setMinimumWidth(95)
        self.setMaximumWidth(115)


class VolumeCanvas(QWidget):
    """Central 3D view: vispy volume flanked by segmentation-style LUT bars."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._vispy_ready = False
        self._canvas = None
        self._view = None
        self._image_node = None
        self._label_node = None
        self._blend = 0.5
        self._voxel_dz = 1.0
        self._voxel_dy = 1.0
        self._voxel_dx = 1.0
        self._pending_fit = False
        self._hidden_label_ids: set[int] = set()

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._image_lut = VolumeImageLutBar()
        self._image_lut.gradient.sigGradientChanged.connect(self._on_image_lut_changed)
        self._label_lut = VolumeLabelLutBar()
        self._label_lut.gradient.sigGradientChanged.connect(self._on_label_lut_changed)

        self._left_lut = _LutColumn(self._image_lut)
        self._canvas_host = QWidget()
        self._canvas_host.setMinimumSize(480, 360)
        self._canvas_host.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._right_lut = _LutColumn(self._label_lut)

        layout.addWidget(self._left_lut)
        layout.addWidget(self._canvas_host, stretch=1)
        layout.addWidget(self._right_lut)

        self._ensure_vispy()

    def set_blend(self, value_0_to_100: int) -> None:
        self._blend = max(0.0, min(100.0, float(value_0_to_100))) / 100.0
        self._apply_blend()

    def set_voxel_sizes(self, dz: float, dy: float, dx: float) -> None:
        """Apply physical voxel sizes from metadata (Cell-ACDC ``PhysicalSize*``)."""
        self._voxel_dz = float(dz)
        self._voxel_dy = float(dy)
        self._voxel_dx = float(dx)
        self._apply_voxel_scale()

    def set_volumes(
        self,
        image_volume: np.ndarray,
        label_volume: np.ndarray | None,
        *,
        label_lut_size: int,
    ) -> None:
        self._ensure_vispy()
        assert self._image_node is not None
        assert self._label_node is not None
        assert self._view is not None
        assert self._canvas is not None

        self._label_lut.set_lut_size(label_lut_size)

        self._image_node.set_data(np.ascontiguousarray(image_volume, dtype=np.float32))
        self._apply_image_style()

        if label_volume is not None and np.any(label_volume):
            self._label_node.set_data(np.ascontiguousarray(label_volume, dtype=np.float32))
            self._apply_label_style()
            self._label_node.visible = True
            self._schedule_fit_camera()
        else:
            self._label_node.visible = False

        self._apply_voxel_scale()
        self._apply_blend()
        self._canvas.update()

    def set_hidden_labels(self, hidden_ids: set[int]) -> None:
        """Toggle label visibility via LUT alpha (no volume re-upload)."""
        self._hidden_label_ids = set(hidden_ids)
        if self._label_node is None or not self._label_node.visible:
            return
        self._apply_label_style()
        if self._canvas is not None:
            self._canvas.update()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._pending_fit:
            self._schedule_fit_camera()

    def _ensure_vispy(self) -> None:
        if self._vispy_ready:
            return
        import os

        import vispy
        from vispy import gloo, scene
        from vispy.scene import visuals

        os.environ.setdefault("QT_API", "pyside6")
        vispy.use(app="pyside6")

        self._canvas = scene.SceneCanvas(
            keys="interactive",
            bgcolor="black",
            show=False,
            decorate=False,
        )
        native = self._canvas.native
        native.setFocusPolicy(Qt.StrongFocus)
        native.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        host_layout = QVBoxLayout(self._canvas_host)
        host_layout.setContentsMargins(0, 0, 0, 0)
        host_layout.setSpacing(0)
        host_layout.addWidget(native)

        self._view = self._canvas.central_widget.add_view()
        self._view.camera = scene.cameras.TurntableCamera(
            fov=45,
            elevation=30.0,
            azimuth=-60.0,
        )
        self._image_node = visuals.Volume(np.zeros((2, 2, 2), dtype=np.float32), parent=self._view.scene)
        self._label_node = visuals.Volume(
            np.zeros((2, 2, 2), dtype=np.float32),
            method="translucent",
            interpolation="nearest",
            parent=self._view.scene,
        )
        self._label_node.visible = False
        gloo.set_state(blend=True, depth_test=False)
        self._vispy_ready = True

    def _schedule_fit_camera(self) -> None:
        self._pending_fit = True
        QTimer.singleShot(0, self._fit_camera)

    def _fit_camera(self) -> None:
        if self._view is None or self._view.camera is None:
            return
        self._pending_fit = False
        self._view.camera.set_range()
        self._view.camera.elevation = 30.0
        self._view.camera.azimuth = -60.0
        if self._canvas is not None:
            self._canvas.update()

    def _apply_voxel_scale(self, node=None) -> None:
        if not self._vispy_ready:
            return
        from vispy.visuals.transforms import STTransform

        scale = voxel_display_scale(self._voxel_dz, self._voxel_dy, self._voxel_dx)
        transform = STTransform(scale=scale)
        targets = [node] if node is not None else [self._image_node, self._label_node]
        for target in targets:
            if target is not None:
                target.transform = transform
        if self._canvas is not None:
            self._canvas.update()

    def _apply_image_style(self) -> None:
        assert self._image_node is not None
        self._image_node.cmap = pg_colormap_to_vispy(self._image_lut.gradient.colorMap())
        self._image_node.clim = (0.0, 1.0)

    def _apply_label_style(self) -> None:
        assert self._label_node is not None
        lut = self._label_lut.rgba_lut(self._hidden_label_ids)
        self._label_node.cmap = label_lut_to_vispy(lut)
        n = float(self._label_lut.lut_size)
        self._label_node.clim = (-0.5, n - 0.5)

    def _apply_blend(self) -> None:
        t = self._blend
        if self._image_node is not None:
            self._image_node.opacity = 1.0 - t
        if self._label_node is not None and self._label_node.visible:
            self._label_node.opacity = t
        if self._canvas is not None:
            self._canvas.update()

    def _on_image_lut_changed(self) -> None:
        if self._image_node is not None:
            self._apply_image_style()
            if self._canvas is not None:
                self._canvas.update()

    def _on_label_lut_changed(self) -> None:
        if self._label_node is not None and self._label_node.visible:
            self._apply_label_style()
            if self._canvas is not None:
                self._canvas.update()

    def focus_canvas(self) -> None:
        self._ensure_vispy()
        if self._canvas is not None:
            self._canvas.native.setFocus()
