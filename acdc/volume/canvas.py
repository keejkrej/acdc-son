"""Vispy volume canvas with dual LUT bars."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import QHBoxLayout, QSizePolicy, QVBoxLayout, QWidget

from acdc.blend import layer_opacities
from acdc.volume.cmaps import label_lut_to_vispy, pg_colormap_to_vispy
from acdc.volume.gl_blend import volume_gl_state
from acdc.volume.lut import VolumeImageLutBar, VolumeLabelLutBar
from acdc.volume.prepare import voxel_display_scale


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
        self._secondary_node = None
        self._primary_secondary_blend = 50.0
        self._image_seg_blend = 50.0
        self._has_secondary = False
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
        self._image_lut.sigLevelsChanged.connect(self._on_image_lut_changed)
        self._secondary_lut = VolumeImageLutBar(axis_label="Secondary")
        self._secondary_lut.gradient.sigGradientChanged.connect(self._on_secondary_lut_changed)
        self._secondary_lut.sigLevelsChanged.connect(self._on_secondary_lut_changed)
        self._label_lut = VolumeLabelLutBar()
        self._label_lut.gradient.sigGradientChanged.connect(self._on_label_lut_changed)

        self._left_stack = QWidget()
        left_layout = QHBoxLayout(self._left_stack)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)
        self._image_lut_column = _LutColumn(self._image_lut)
        self._secondary_lut_column = _LutColumn(self._secondary_lut)
        self._secondary_lut_column.setVisible(False)
        left_layout.addWidget(self._image_lut_column, stretch=1)
        left_layout.addWidget(self._secondary_lut_column, stretch=1)
        self._left_stack.setMinimumWidth(190)
        self._left_stack.setMaximumWidth(230)

        self._canvas_host = QWidget()
        self._canvas_host.setMinimumSize(480, 360)
        self._canvas_host.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._right_lut = _LutColumn(self._label_lut)

        layout.addWidget(self._left_stack)
        layout.addWidget(self._canvas_host, stretch=1)
        layout.addWidget(self._right_lut)

        self._ensure_vispy()

    def set_primary_secondary_blend(self, value_0_to_100: int) -> None:
        self._primary_secondary_blend = max(0.0, min(100.0, float(value_0_to_100)))
        self._apply_layer_blend()

    def set_image_seg_blend(self, value_0_to_100: int) -> None:
        self._image_seg_blend = max(0.0, min(100.0, float(value_0_to_100)))
        self._apply_layer_blend()

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
        image_clim: tuple[float, float] = (0.0, 1.0),
    ) -> None:
        self._ensure_vispy()
        assert self._image_node is not None
        assert self._label_node is not None
        assert self._view is not None
        assert self._canvas is not None

        self._label_lut.set_lut_size(label_lut_size)

        self._image_node.set_data(np.ascontiguousarray(image_volume, dtype=np.float32))
        self._image_lut.setLevels(*image_clim)
        self._apply_image_style()

        if label_volume is not None and np.any(label_volume):
            self._label_node.set_data(np.ascontiguousarray(label_volume, dtype=np.float32))
            self._apply_label_style()
            self._label_node.visible = True
            self._schedule_fit_camera()
        else:
            self._label_node.visible = False

        self._apply_voxel_scale()
        self._apply_layer_blend()
        self._canvas.update()

    def set_secondary_volume(
        self,
        volume: np.ndarray,
        *,
        clim: tuple[float, float] = (0.0, 1.0),
    ) -> None:
        self._ensure_vispy()
        assert self._secondary_node is not None
        assert self._canvas is not None

        self._secondary_node.set_data(np.ascontiguousarray(volume, dtype=np.float32))
        self._secondary_lut.setLevels(*clim)
        self._apply_secondary_style()
        self._secondary_node.visible = True
        self._has_secondary = True
        self._secondary_lut_column.setVisible(True)
        self._apply_voxel_scale(self._secondary_node)
        self._apply_layer_blend()
        self._sync_left_lut_width()
        self._schedule_fit_camera()
        self._canvas.update()

    def clear_secondary_volume(self) -> None:
        if self._secondary_node is not None:
            self._secondary_node.visible = False
        self._has_secondary = False
        self._secondary_lut_column.setVisible(False)
        self._sync_left_lut_width()
        self._apply_layer_blend()
        if self._canvas is not None:
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
        from vispy import scene
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
        scene_root = self._view.scene
        self._image_node = visuals.Volume(
            np.zeros((2, 2, 2), dtype=np.float32),
            method="mip",
            parent=scene_root,
        )
        self._secondary_node = visuals.Volume(
            np.zeros((2, 2, 2), dtype=np.float32),
            method="mip",
            parent=scene_root,
        )
        self._secondary_node.visible = False
        self._label_node = visuals.Volume(
            np.zeros((2, 2, 2), dtype=np.float32),
            method="translucent",
            interpolation="nearest",
            parent=scene_root,
        )
        self._label_node.visible = False
        self._image_node.order = 0
        self._secondary_node.order = 1
        self._label_node.order = 2
        self._apply_volume_gl_blend()
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
        if node is not None:
            targets = [node]
        else:
            targets = [self._image_node, self._secondary_node, self._label_node]
        for target in targets:
            if target is not None:
                target.transform = transform
        if self._canvas is not None:
            self._canvas.update()

    def _sync_left_lut_width(self) -> None:
        if self._has_secondary:
            self._left_stack.setMinimumWidth(190)
            self._left_stack.setMaximumWidth(230)
        else:
            self._left_stack.setMinimumWidth(95)
            self._left_stack.setMaximumWidth(115)

    def _clim_from_lut(self, lut_bar: VolumeImageLutBar) -> tuple[float, float]:
        lo, hi = lut_bar.getLevels()
        lo = float(lo)
        hi = float(hi)
        if hi <= lo:
            hi = lo + 1e-6
        return lo, hi

    def _apply_image_style(self) -> None:
        assert self._image_node is not None
        self._image_node.cmap = pg_colormap_to_vispy(self._image_lut.gradient.colorMap())
        self._image_node.clim = self._clim_from_lut(self._image_lut)

    def _apply_label_style(self) -> None:
        assert self._label_node is not None
        lut = self._label_lut.rgba_lut(self._hidden_label_ids)
        self._label_node.cmap = label_lut_to_vispy(lut)
        n = float(self._label_lut.lut_size)
        self._label_node.clim = (-0.5, n - 0.5)

    def _apply_secondary_style(self) -> None:
        assert self._secondary_node is not None
        self._secondary_node.cmap = pg_colormap_to_vispy(self._secondary_lut.gradient.colorMap())
        self._secondary_node.clim = self._clim_from_lut(self._secondary_lut)

    def _apply_layer_blend(self) -> None:
        primary_opacity, secondary_opacity, seg_opacity = layer_opacities(
            self._primary_secondary_blend,
            self._image_seg_blend,
            has_secondary=self._has_secondary,
        )
        if self._image_node is not None:
            self._image_node.opacity = primary_opacity
        if self._secondary_node is not None and self._secondary_node.visible:
            self._secondary_node.opacity = secondary_opacity
        if self._label_node is not None and self._label_node.visible:
            self._label_node.opacity = seg_opacity
        self._apply_volume_gl_blend()
        if self._canvas is not None:
            self._canvas.update()

    def _apply_volume_gl_blend(self) -> None:
        """Composite fluor MIP volumes like napari (translucent + additive)."""
        if not self._vispy_ready:
            return
        assert self._image_node is not None
        assert self._secondary_node is not None
        assert self._label_node is not None

        def _active(node, opacity: float) -> bool:
            return bool(node.visible) and opacity > 1e-6

        primary_opacity = float(self._image_node.opacity)
        secondary_opacity = (
            float(self._secondary_node.opacity)
            if self._has_secondary and self._secondary_node.visible
            else 0.0
        )
        primary_active = _active(self._image_node, primary_opacity)
        secondary_active = _active(self._secondary_node, secondary_opacity)

        # First contributing fluor layer uses napari's bottom-layer blend func.
        self._image_node.set_gl_state(
            **volume_gl_state(
                "translucent_no_depth",
                first_visible=primary_active,
            )
        )
        if self._has_secondary:
            self._secondary_node.set_gl_state(
                **volume_gl_state(
                    "additive",
                    first_visible=secondary_active and not primary_active,
                )
            )

        self._label_node.set_gl_state(
            **volume_gl_state("translucent", first_visible=False)
        )

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

    def _on_secondary_lut_changed(self) -> None:
        if self._secondary_node is not None and self._secondary_node.visible:
            self._apply_secondary_style()
            if self._canvas is not None:
                self._canvas.update()

    def focus_canvas(self) -> None:
        self._ensure_vispy()
        if self._canvas is not None:
            self._canvas.native.setFocus()
