"""Vispy volume canvas with per-channel LUT bars."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import QHBoxLayout, QSizePolicy, QVBoxLayout, QWidget

from acdc.blend import display_opacities
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


@dataclass
class _ImageChannel:
    node: object
    lut: VolumeImageLutBar
    column: _LutColumn


class VolumeCanvas(QWidget):
    """Central 3D view: vispy volumes flanked by per-channel and label LUT bars."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._vispy_ready = False
        self._canvas = None
        self._view = None
        self._scene_root = None
        self._label_node = None
        self._image_channels: list[_ImageChannel] = []
        self._channel_weights: list[float] = [1.0]
        self._image_seg_blend = 50.0
        self._voxel_dz = 1.0
        self._voxel_dy = 1.0
        self._voxel_dx = 1.0
        self._pending_fit = False
        self._hidden_label_ids: set[int] = set()

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._label_lut = VolumeLabelLutBar()
        self._label_lut.gradient.sigGradientChanged.connect(self._on_label_lut_changed)

        self._left_stack = QWidget()
        self._left_layout = QHBoxLayout(self._left_stack)
        self._left_layout.setContentsMargins(0, 0, 0, 0)
        self._left_layout.setSpacing(0)
        self._left_stack.setMinimumWidth(95)
        self._left_stack.setMaximumWidth(115)

        self._canvas_host = QWidget()
        self._canvas_host.setMinimumSize(480, 360)
        self._canvas_host.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._right_lut = _LutColumn(self._label_lut)

        layout.addWidget(self._left_stack)
        layout.addWidget(self._canvas_host, stretch=1)
        layout.addWidget(self._right_lut)

        self._ensure_vispy()

    def set_channel_weights(self, weights: list[float]) -> None:
        self._channel_weights = [max(0.0, min(1.0, float(w))) for w in weights]
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

    def set_image_channels(
        self,
        volumes: list[np.ndarray],
        *,
        clims: list[tuple[float, float]],
        label_volume: np.ndarray | None,
        label_lut_size: int,
        channel_labels: list[str] | None = None,
    ) -> None:
        self._ensure_vispy()
        assert self._label_node is not None
        assert self._view is not None
        assert self._canvas is not None

        if len(volumes) != len(clims):
            raise ValueError("volumes and clims must have the same length")

        labels = channel_labels or []
        self._sync_image_channel_count(len(volumes), labels)

        self._label_lut.set_lut_size(label_lut_size)
        for index, (volume, clim) in enumerate(zip(volumes, clims, strict=True)):
            channel = self._image_channels[index]
            channel.node.set_data(np.ascontiguousarray(volume, dtype=np.float32))
            channel.lut.setLevels(*clim)
            self._apply_channel_style(index)

        if label_volume is not None and np.any(label_volume):
            self._label_node.set_data(np.ascontiguousarray(label_volume, dtype=np.float32))
            self._apply_label_style()
            self._label_node.visible = True
            self._schedule_fit_camera()
        else:
            self._label_node.visible = False

        if len(self._channel_weights) != len(volumes):
            self._channel_weights = [1.0] * len(volumes)

        self._apply_voxel_scale()
        self._apply_layer_blend()
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

    def _sync_image_channel_count(self, count: int, labels: list[str]) -> None:
        self._ensure_vispy()
        assert self._scene_root is not None

        while len(self._image_channels) > count:
            channel = self._image_channels.pop()
            channel.node.parent = None
            channel.node = None
            self._left_layout.removeWidget(channel.column)
            channel.column.deleteLater()

        while len(self._image_channels) < count:
            index = len(self._image_channels)
            label = labels[index] if index < len(labels) else f"Channel {index + 1}"
            from vispy.scene import visuals

            node = visuals.Volume(
                np.zeros((2, 2, 2), dtype=np.float32),
                method="mip",
                parent=self._scene_root,
            )
            lut = VolumeImageLutBar(axis_label=label)
            lut.gradient.sigGradientChanged.connect(
                lambda _=None, i=index: self._on_channel_lut_changed(i)
            )
            lut.sigLevelsChanged.connect(lambda _=None, i=index: self._on_channel_lut_changed(i))
            column = _LutColumn(lut)
            self._left_layout.addWidget(column)
            self._image_channels.append(_ImageChannel(node=node, lut=lut, column=column))

        for index, channel in enumerate(self._image_channels):
            channel.node.order = index
            channel.column.setVisible(True)

        if self._label_node is not None:
            self._label_node.order = count
        self._sync_left_lut_width()

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
        self._scene_root = self._view.scene
        self._label_node = visuals.Volume(
            np.zeros((2, 2, 2), dtype=np.float32),
            method="translucent",
            interpolation="nearest",
            parent=self._scene_root,
        )
        self._label_node.visible = False
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

    def _apply_voxel_scale(self) -> None:
        if not self._vispy_ready:
            return
        from vispy.visuals.transforms import STTransform

        scale = voxel_display_scale(self._voxel_dz, self._voxel_dy, self._voxel_dx)
        transform = STTransform(scale=scale)
        for channel in self._image_channels:
            channel.node.transform = transform
        if self._label_node is not None:
            self._label_node.transform = transform
        if self._canvas is not None:
            self._canvas.update()

    def _sync_left_lut_width(self) -> None:
        n = max(1, len(self._image_channels))
        width = min(115 * n, 460)
        self._left_stack.setMinimumWidth(width)
        self._left_stack.setMaximumWidth(width + 20)

    def _clim_from_lut(self, lut_bar: VolumeImageLutBar) -> tuple[float, float]:
        lo, hi = lut_bar.getLevels()
        lo = float(lo)
        hi = float(hi)
        if hi <= lo:
            hi = lo + 1e-6
        return lo, hi

    def _apply_channel_style(self, index: int) -> None:
        channel = self._image_channels[index]
        channel.node.cmap = pg_colormap_to_vispy(channel.lut.gradient.colorMap())
        channel.node.clim = self._clim_from_lut(channel.lut)

    def _apply_label_style(self) -> None:
        assert self._label_node is not None
        lut = self._label_lut.rgba_lut(self._hidden_label_ids)
        self._label_node.cmap = label_lut_to_vispy(lut)
        n = float(self._label_lut.lut_size)
        self._label_node.clim = (-0.5, n - 0.5)

    def _apply_layer_blend(self) -> None:
        channel_opacities, seg_opacity = display_opacities(
            self._channel_weights,
            self._image_seg_blend,
        )
        for index, channel in enumerate(self._image_channels):
            opacity = channel_opacities[index] if index < len(channel_opacities) else 0.0
            channel.node.opacity = opacity
            channel.node.visible = opacity > 1e-6
        if self._label_node is not None and self._label_node.visible:
            self._label_node.opacity = seg_opacity
        self._apply_volume_gl_blend()
        if self._canvas is not None:
            self._canvas.update()

    def _apply_volume_gl_blend(self) -> None:
        """Composite fluor MIP volumes like napari (translucent + additive)."""
        if not self._vispy_ready:
            return
        assert self._label_node is not None

        def _active(node, opacity: float) -> bool:
            return bool(node.visible) and opacity > 1e-6

        first_active_set = False
        for channel in self._image_channels:
            opacity = float(channel.node.opacity)
            active = _active(channel.node, opacity)
            first_visible = active and not first_active_set
            if first_visible:
                first_active_set = True
            blending = "translucent_no_depth" if first_visible else "additive"
            channel.node.set_gl_state(
                **volume_gl_state(blending, first_visible=first_visible)
            )

        self._label_node.set_gl_state(
            **volume_gl_state("translucent", first_visible=False)
        )

    def _on_channel_lut_changed(self, index: int) -> None:
        if index >= len(self._image_channels):
            return
        channel = self._image_channels[index]
        if channel.node.visible:
            self._apply_channel_style(index)
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
