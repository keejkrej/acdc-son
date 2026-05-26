"""Segmentation view: Qt widgets and pyqtgraph display (no business logic)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from pyqtgraph import Point
from pyqtgraph import functions as fn
from qtpy.QtCore import QEvent, Qt, Signal
from qtpy.QtGui import QActionGroup, QCloseEvent
from qtpy.QtWidgets import (
    QAbstractItemView,
    QAction,
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QListWidget,
    QListWidgetItem,
    QSlider,
    QSpinBox,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from acdc.utils.blend import display_opacities
from acdc.ui.blend_controls import BlendControlBar
from acdc.ui.dialogs import pick_from_list, pick_many_from_list

from acdc.utils.display_levels import autoscale_levels

from .lut import ImageLutBar, SegmentationLutBar
from . import editing
from acdc.ui.icons import LucideIcon, install_icon_theme_watcher, themed_lucide_qicon


@dataclass
class _ChannelSlot:
    image_item: pg.ImageItem
    lut: ImageLutBar
    display_levels: tuple[float, float] | None = None


class ImageCanvas(QWidget):
    """Pyqtgraph canvas that emits image coordinates on paint gestures."""

    paint_at = Signal(int, int)  # y, x
    stroke_started = Signal()
    stroke_finished = Signal()
    brush_size_changed = Signal(int)
    pick_at = Signal(int, int)  # y, x
    rect_pick = Signal(int, int, int, int)  # y0, x0, y1, x1

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._layout = pg.GraphicsLayoutWidget()
        layout.addWidget(self._layout)
        self._plot = self._layout.addPlot(row=0, col=1)
        self._plot.invertY(True)
        self._plot.setAspectLocked(True)
        self._plot.hideAxis("bottom")
        self._plot.hideAxis("left")
        self._channel_slots: list[_ChannelSlot] = []
        self._mask_item = pg.ImageItem()
        self._contour_item = pg.ImageItem()
        self._contour_labels: np.ndarray | None = None
        self._plot.addItem(self._mask_item)
        self._plot.addItem(self._contour_item)
        self._mask_item.setZValue(10)
        self._contour_item.setZValue(11)
        self._seg_lut = SegmentationLutBar(self._mask_item)
        first_item = pg.ImageItem()
        first_lut = ImageLutBar(first_item, axis_label="Image")
        first_lut.gradient.sigGradientChanged.connect(self._apply_layer_blend)
        first_item.setZValue(1)
        self._plot.addItem(first_item)
        self._channel_slots.append(
            _ChannelSlot(image_item=first_item, lut=first_lut, display_levels=None)
        )
        self._layout.addItem(first_lut, row=0, col=0)
        self._layout.addItem(self._seg_lut, row=0, col=2)
        self._tool = "hand"
        self._brush_size = 4
        self._space_pan = False
        self._resize_start_x = 0
        self._resize_start_size = 4
        self._stroke_armed = False
        self._select_drag_start: tuple[int, int] | None = None
        self._marquee_item: pg.PlotDataItem | None = None
        self._selection_items: list[pg.PlotDataItem] = []
        self._viewbox = self._plot.getViewBox()
        self._viewbox.setMenuEnabled(False)
        self._orig_drag = self._viewbox.mouseDragEvent
        self._orig_click = self._viewbox.mouseClickEvent
        self._viewbox.mouseDragEvent = self._mouse_drag_event  # type: ignore[method-assign]
        self._viewbox.mouseClickEvent = self._mouse_click_event  # type: ignore[method-assign]
        scene = self._layout.scene()
        assert scene is not None
        scene.sigMouseMoved.connect(self._on_scene_mouse_moved)
        self.setFocusPolicy(Qt.StrongFocus)
        self._layout.setFocusPolicy(Qt.StrongFocus)
        self._layout.installEventFilter(self)
        self._seg_lut.gradient.sigGradientChanged.connect(self._on_seg_lut_changed)
        self._source_labels: np.ndarray | None = None
        self._hidden_label_ids: set[int] = set()
        self._channel_weights: list[float] = [1.0]
        self._image_seg_blend = 50.0
        self._mask_max_label_id = 0
        self._seg_display_max = -1

    @property
    def _image_item(self) -> pg.ImageItem:
        return self._channel_slots[0].image_item

    def set_channel_weights(self, weights: list[float]) -> None:
        self._channel_weights = [max(0.0, min(1.0, float(w))) for w in weights]
        self._apply_layer_blend()

    def set_image_seg_blend(self, value_0_to_100: float) -> None:
        self._image_seg_blend = max(0.0, min(100.0, float(value_0_to_100)))
        self._apply_layer_blend()

    def _apply_layer_blend(self) -> None:
        channel_opacities, seg_opacity = display_opacities(
            self._channel_weights,
            self._image_seg_blend,
        )
        for index, slot in enumerate(self._channel_slots):
            opacity = channel_opacities[index] if index < len(channel_opacities) else 0.0
            slot.image_item.setVisible(opacity > 1e-6)
            slot.image_item.setOpacity(opacity)
            slot.image_item.update()
        self._mask_item.setOpacity(seg_opacity)
        self._mask_item.update()
        scene = self._plot.scene()
        if scene is not None:
            scene.update()

    def set_mask_max_label_id(self, max_label_id: int) -> None:
        self._mask_max_label_id = max(int(max_label_id), 0)

    def set_channel_slices(
        self,
        slices: list[np.ndarray],
        *,
        display_levels: list[tuple[float, float] | None] | None = None,
        labels: list[str] | None = None,
    ) -> None:
        if display_levels is None:
            display_levels = [None] * len(slices)
        if labels is None:
            labels = [f"Channel {i + 1}" for i in range(len(slices))]
        self._sync_channel_count(len(slices), labels)
        if len(self._channel_weights) != len(slices):
            self._channel_weights = [1.0] * len(slices)
        for index, gray in enumerate(slices):
            levels = display_levels[index] if index < len(display_levels) else None
            self._channel_slots[index].display_levels = levels
            self._set_channel_image(index, np.asarray(gray), levels)
        self._apply_layer_blend()

    def _sync_channel_count(self, count: int, labels: list[str]) -> None:
        while len(self._channel_slots) > count:
            slot = self._channel_slots.pop()
            self._plot.removeItem(slot.image_item)
            self._layout.removeItem(slot.lut)
            slot.lut.hide()

        while len(self._channel_slots) < count:
            index = len(self._channel_slots)
            label = labels[index] if index < len(labels) else f"Channel {index + 1}"
            image_item = pg.ImageItem()
            lut = ImageLutBar(image_item, axis_label=label)
            lut.gradient.sigGradientChanged.connect(self._apply_layer_blend)
            image_item.setZValue(index + 1)
            self._plot.addItem(image_item)
            self._channel_slots.append(
                _ChannelSlot(image_item=image_item, lut=lut, display_levels=None)
            )

        self._rebuild_lut_columns()

    def _rebuild_lut_columns(self) -> None:
        for slot in list(self._channel_slots):
            try:
                self._layout.removeItem(slot.lut)
            except ValueError:
                pass
        try:
            self._layout.removeItem(self._plot)
        except ValueError:
            pass
        try:
            self._layout.removeItem(self._seg_lut)
        except ValueError:
            pass
        for index, slot in enumerate(self._channel_slots):
            self._layout.addItem(slot.lut, row=0, col=index)
            slot.lut.show()
        plot_col = len(self._channel_slots)
        self._layout.addItem(self._plot, row=0, col=plot_col)
        self._layout.addItem(self._seg_lut, row=0, col=plot_col + 1)

    def _set_channel_image(
        self,
        index: int,
        gray: np.ndarray,
        display_levels: tuple[float, float] | None,
    ) -> None:
        slot = self._channel_slots[index]
        shape_changed = (
            slot.image_item.image is None or slot.image_item.image.shape != gray.shape
        )
        if display_levels is not None:
            lo, hi = display_levels
            if shape_changed:
                slot.image_item.setImage(gray, autoLevels=False, levels=(lo, hi))
                slot.lut.setLevels(lo, hi)
                slot.lut.imageChanged(autoLevel=False, autoRange=True)
            else:
                slot.image_item.setImage(gray, autoLevels=False, levels=(lo, hi))
        elif shape_changed:
            lo, hi = autoscale_levels(gray)
            slot.image_item.setImage(gray, autoLevels=False, levels=(lo, hi))
            slot.lut.setLevels(lo, hi)
            slot.lut.imageChanged(autoLevel=False, autoRange=True)
        else:
            slot.image_item.setImage(gray, autoLevels=False)

    def set_mask_labels(self, labels: np.ndarray) -> None:
        self._source_labels = labels
        self._render_mask_labels(upload=True)

    def set_hidden_labels(self, hidden_ids: set[int]) -> None:
        """Toggle label visibility via LUT alpha (mask stays indexed by raw IDs)."""
        self._hidden_label_ids = set(hidden_ids)
        self._render_mask_labels(upload=False)

    def _render_mask_labels(self, *, upload: bool) -> None:
        labels = self._source_labels
        if labels is None:
            return
        max_label = max(int(labels.max(initial=0)), self._mask_max_label_id)
        display_max = max(max_label, 1)
        if display_max != self._seg_display_max:
            self._seg_display_max = display_max
            self._seg_lut.set_label_display_max(max_label)

        current = self._mask_item.image
        if upload or current is None or current.shape != labels.shape:
            self._mask_item.setImage(
                labels,
                autoLevels=False,
                levels=(0, display_max),
            )
        self._apply_mask_lut()
        self._update_contours(self._labels_for_contours())
        self._apply_layer_blend()

    def _apply_mask_lut(self) -> None:
        from acdc.segment.lut import lut_with_hidden_labels

        lut = lut_with_hidden_labels(self._seg_lut.current_lut(), self._hidden_label_ids)
        self._mask_item.setLookupTable(lut)

    def _labels_for_contours(self) -> np.ndarray:
        labels = self._source_labels
        if labels is None:
            return np.array([], dtype=np.uint32)
        if not self._hidden_label_ids:
            return labels
        return editing.apply_label_visibility(labels, self._hidden_label_ids)

    def _on_seg_lut_changed(self) -> None:
        self._apply_mask_lut()
        if self._source_labels is not None:
            self._update_contours(self._labels_for_contours())

    def _update_contours(self, labels: np.ndarray) -> None:
        self._contour_labels = labels
        if labels.size == 0 or not np.any(labels):
            self._contour_item.clear()
            return
        rgba = editing.labels_to_contour_rgba(
            labels,
            self._seg_lut.current_lut(),
            thickness=1,
        )
        self._contour_item.setImage(rgba, autoLevels=False)

    def clear(self) -> None:
        for slot in self._channel_slots:
            slot.image_item.clear()
        self._mask_item.clear()
        self._contour_item.clear()
        self._contour_labels = None
        self._source_labels = None
        self._hidden_label_ids = set()
        self._channel_weights = [1.0]
        self._mask_max_label_id = 0
        self._seg_display_max = -1
        self._clear_marquee()
        self._clear_selection_items()

    def set_selection(self, label_ids: list[int], mask_slice: np.ndarray) -> None:
        """Draw dashed bounding boxes for the selected label IDs."""
        self._clear_selection_items()
        if not label_ids:
            return
        pen = pg.mkPen((255, 220, 0), width=2, style=Qt.DashLine)
        for label_id in label_ids:
            box = editing.label_bounding_box(mask_slice, label_id)
            if box is None:
                continue
            ymin, xmin, ymax, xmax = box
            xs = [xmin, xmax, xmax, xmin, xmin]
            ys = [ymin, ymin, ymax, ymax, ymin]
            item = pg.PlotDataItem(xs, ys, pen=pen)
            item.setZValue(12)
            self._plot.addItem(item)
            self._selection_items.append(item)

    def _clear_selection_items(self) -> None:
        for item in self._selection_items:
            self._plot.removeItem(item)
        self._selection_items.clear()

    def _clear_marquee(self) -> None:
        if self._marquee_item is not None:
            self._plot.removeItem(self._marquee_item)
            self._marquee_item = None

    def _set_marquee(self, y0: int, x0: int, y1: int, x1: int) -> None:
        xmin, xmax = sorted((x0, x1))
        ymin, ymax = sorted((y0, y1))
        xs = [xmin, xmax, xmax, xmin, xmin]
        ys = [ymin, ymin, ymax, ymax, ymin]
        pen = pg.mkPen((255, 220, 0), width=1, style=Qt.DashLine)
        if self._marquee_item is None:
            self._marquee_item = pg.PlotDataItem(pen=pen)
            self._marquee_item.setZValue(12)
            self._plot.addItem(self._marquee_item)
        self._marquee_item.setData(xs, ys)

    def set_tool(self, tool: str) -> None:
        self._tool = tool
        self._finish_stroke_if_armed()
        self._select_drag_start = None
        self._clear_marquee()

    def set_brush_size(self, size: int) -> None:
        self._brush_size = size

    def _finish_stroke_if_armed(self) -> None:
        if not self._stroke_armed:
            return
        self._stroke_armed = False
        self.stroke_finished.emit()

    def _paint_tool_active(self) -> bool:
        return self._tool in {"brush", "eraser"}

    def _emit_paint_at(self, scene_pos) -> None:
        mapped = self._map_pos_to_pixel(scene_pos)
        if mapped is not None:
            self.paint_at.emit(*mapped)

    def eventFilter(self, watched, event) -> bool:  # noqa: ARG002
        if event.type() == QEvent.KeyPress:
            if event.key() == Qt.Key_Space and not event.isAutoRepeat():
                self._space_pan = True
        elif event.type() == QEvent.KeyRelease:
            if event.key() == Qt.Key_Space and not event.isAutoRepeat():
                self._space_pan = False
                self._finish_stroke_if_armed()
        return False

    def mousePressEvent(self, event) -> None:
        self.setFocus()
        self._layout.setFocus()
        super().mousePressEvent(event)

    def _should_pan(self, button: Qt.MouseButton) -> bool:
        if button == Qt.MiddleButton:
            return True
        if self._space_pan and button == Qt.LeftButton:
            return True
        if self._tool == "hand" and button in (Qt.LeftButton, Qt.RightButton):
            return True
        return False

    def _pan_by_drag(self, ev, axis=None) -> None:
        dif = ev.pos() - ev.lastPos()
        dif = dif * -1
        mouse_enabled = self._viewbox.state["mouseEnabled"]
        mask = [float(v) for v in mouse_enabled]
        if axis is not None:
            mask[1 - axis] = 0.0
        tr = fn.invertQTransform(self._viewbox.childGroup.transform())
        mapped = tr.map(dif * mask) - tr.map(Point(0, 0))
        self._viewbox.translateBy(
            x=mapped.x() if mask[0] else None,
            y=mapped.y() if mask[1] else None,
        )

    def _right_drag_brush_size(self, ev) -> None:
        if ev.isStart():
            self._resize_start_x = int(ev.screenPos().x())
            self._resize_start_size = self._brush_size
            return
        dx = int(ev.screenPos().x()) - self._resize_start_x
        new_size = max(1, min(128, self._resize_start_size + dx // 3))
        if new_size != self._brush_size:
            self._brush_size = new_size
            self.brush_size_changed.emit(new_size)

    def _map_pos_to_pixel(self, pos) -> tuple[int, int] | None:
        img = self._image_item.image
        if img is None:
            return None
        mouse_point = self._viewbox.mapSceneToView(pos)
        x = int(round(mouse_point.x()))
        y = int(round(mouse_point.y()))
        h, w = img.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            return y, x
        return None

    def _toggle_lazy_stroke(self, scene_pos) -> None:
        if self._stroke_armed:
            self._emit_paint_at(scene_pos)
            self._finish_stroke_if_armed()
            return
        if self._map_pos_to_pixel(scene_pos) is None:
            return
        self._stroke_armed = True
        self.stroke_started.emit()
        self._emit_paint_at(scene_pos)

    def _mouse_drag_event(self, ev, axis=None) -> None:
        button = ev.button()

        if button == Qt.RightButton and self._tool in {"brush", "eraser"}:
            self._right_drag_brush_size(ev)
            ev.accept()
            return

        if self._should_pan(button):
            self._finish_stroke_if_armed()
            if self._tool == "hand" and button == Qt.RightButton:
                self._pan_by_drag(ev, axis)
            else:
                self._orig_drag(ev, axis)
            ev.accept()
            return

        if button != Qt.LeftButton:
            self._orig_drag(ev, axis)
            return

        if self._paint_tool_active():
            if ev.isStart():
                self._toggle_lazy_stroke(ev.scenePos())
            elif self._stroke_armed:
                self._emit_paint_at(ev.scenePos())
            ev.accept()
            return

        if self._tool == "move" and button == Qt.LeftButton:
            if ev.isStart():
                self._select_drag_start = self._map_pos_to_pixel(ev.scenePos())
                ev.accept()
                return
            if self._select_drag_start is not None:
                end = self._map_pos_to_pixel(ev.scenePos())
                y0, x0 = self._select_drag_start
                if ev.isFinish():
                    self._clear_marquee()
                    if end is None:
                        self.pick_at.emit(y0, x0)
                    elif y0 == end[0] and x0 == end[1]:
                        self.pick_at.emit(y0, x0)
                    else:
                        self.rect_pick.emit(y0, x0, end[0], end[1])
                    self._select_drag_start = None
                elif end is not None:
                    self._set_marquee(y0, x0, end[0], end[1])
            ev.accept()
            return

        ev.accept()

    def _mouse_click_event(self, ev) -> None:
        if ev.button() == Qt.LeftButton and self._paint_tool_active():
            self._toggle_lazy_stroke(ev.scenePos())
            ev.accept()
            return
        self._orig_click(ev)

    def _on_scene_mouse_moved(self, scene_pos) -> None:
        if self._stroke_armed and self._paint_tool_active():
            self._emit_paint_at(scene_pos)


class TransportBar(QWidget):
    """Bottom viewer transport for frame and Z-slice scrubbing."""

    t_index_changed = Signal(int)
    z_index_changed = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(2)

        self._frame_row = QWidget()
        frame_layout = QHBoxLayout(self._frame_row)
        frame_layout.setContentsMargins(0, 0, 0, 0)
        frame_layout.addWidget(QLabel("Frame:"))
        self._t_slider = QSlider(Qt.Horizontal)
        self._t_slider.valueChanged.connect(self.t_index_changed.emit)
        frame_layout.addWidget(self._t_slider, stretch=1)
        self._t_label = QLabel("0/0")
        self._t_label.setMinimumWidth(48)
        frame_layout.addWidget(self._t_label)
        layout.addWidget(self._frame_row)

        self._z_row = QWidget()
        z_layout = QHBoxLayout(self._z_row)
        z_layout.setContentsMargins(0, 0, 0, 0)
        z_layout.addWidget(QLabel("Z-slice:"))
        self._z_slider = QSlider(Qt.Horizontal)
        self._z_slider.valueChanged.connect(self.z_index_changed.emit)
        z_layout.addWidget(self._z_slider, stretch=1)
        self._z_label = QLabel("0/0")
        self._z_label.setMinimumWidth(48)
        z_layout.addWidget(self._z_label)
        layout.addWidget(self._z_row)

    def set_navigation(self, t: int, t_max: int, z: int, z_max: int) -> None:
        show_frame = t_max > 0
        show_z = z_max > 0
        self.setVisible(show_frame or show_z)
        self._frame_row.setVisible(show_frame)
        self._z_row.setVisible(show_z)

        self._t_slider.blockSignals(True)
        self._z_slider.blockSignals(True)
        self._t_slider.setEnabled(show_frame)
        self._z_slider.setEnabled(show_z)
        self._t_slider.setMinimum(0)
        self._t_slider.setMaximum(max(0, t_max))
        self._z_slider.setMinimum(0)
        self._z_slider.setMaximum(max(0, z_max))
        self._t_slider.setValue(t)
        self._z_slider.setValue(z)
        self._t_slider.blockSignals(False)
        self._z_slider.blockSignals(False)
        self._t_label.setText(f"{t}/{t_max}")
        self._z_label.setText(f"{z}/{z_max}")

    def update_indices(self, t: int, t_max: int, z: int, z_max: int) -> None:
        """Update frame labels without touching slider values (during scrub)."""
        self._t_label.setText(f"{t}/{t_max}")
        self._z_label.setText(f"{z}/{z_max}")


class ViewerFrame(QWidget):
    """Canvas with blend controls and bottom transport."""

    t_index_changed = Signal(int)
    z_index_changed = Signal(int)
    channel_weights_changed = Signal(list)
    image_seg_blend_changed = Signal(int)

    def __init__(self, canvas: ImageCanvas, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.canvas = canvas
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(canvas, stretch=1)
        self._blend_bar = BlendControlBar()
        self._blend_bar.channel_weights_changed.connect(self.channel_weights_changed.emit)
        self._blend_bar.image_seg_changed.connect(self.image_seg_blend_changed.emit)
        self._blend_bar.image_seg_changed.connect(self._on_image_seg_changed)
        self._blend_bar.channel_weights_changed.connect(self._on_channel_weights_changed)
        layout.addWidget(self._blend_bar)
        self._transport = TransportBar()
        self._transport.t_index_changed.connect(self.t_index_changed.emit)
        self._transport.z_index_changed.connect(self.z_index_changed.emit)
        layout.addWidget(self._transport)
        self._on_image_seg_changed(50)
        self._blend_bar.setVisible(False)

    def _on_image_seg_changed(self, value: int) -> None:
        self.canvas.set_image_seg_blend(value)

    def _on_channel_weights_changed(self, weights: list[float]) -> None:
        self.canvas.set_channel_weights(weights)

    def set_blend_controls(
        self,
        *,
        visible: bool,
        channel_names: list[str],
        channel_weights: list[float],
        image_seg: int,
    ) -> None:
        self._blend_bar.setVisible(visible)
        if visible:
            self._blend_bar.set_channels(
                channel_names,
                weights=channel_weights,
                image_seg=image_seg,
            )
            self.canvas.set_channel_weights(channel_weights)
            self.canvas.set_image_seg_blend(image_seg)

    def set_navigation(self, t: int, t_max: int, z: int, z_max: int) -> None:
        self._transport.set_navigation(t, t_max, z, z_max)

    def update_navigation_indices(self, t: int, t_max: int, z: int, z_max: int) -> None:
        self._transport.update_indices(t, t_max, z, z_max)


class LabelListPanel(QWidget):
    """Explorer-style label list with global and per-label visibility toggles."""

    label_id_changed = Signal(int)
    label_visibility_changed = Signal()
    labels_selected = Signal(list)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._updating = False
        self._hidden_labels: set[int] = set()
        self._all_label_ids: list[int] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(0)

        self._list = QListWidget()
        self._list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._list.itemChanged.connect(self._on_item_changed)
        self._list.currentItemChanged.connect(self._on_current_item_changed)
        self._list.itemSelectionChanged.connect(self._on_item_selection_changed)
        layout.addWidget(self._list)

    def reset_visibility(self) -> None:
        self._hidden_labels.clear()

    def get_hidden_label_ids(self) -> set[int]:
        return set(self._hidden_labels)

    def set_label_list(
        self,
        label_ids: list[int],
        *,
        selected_ids: list[int] | None = None,
    ) -> None:
        self._all_label_ids = list(label_ids)
        selected = set(selected_ids or [])

        self._list.blockSignals(True)
        self._updating = True
        self._list.clear()
        for label_id in label_ids:
            item = QListWidgetItem(str(label_id))
            item.setData(Qt.UserRole, label_id)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            checked = label_id not in self._hidden_labels
            item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
            self._list.addItem(item)
        self._updating = False

        self._list.clearSelection()
        for row in range(self._list.count()):
            item = self._list.item(row)
            if item is None:
                continue
            label = item.data(Qt.UserRole)
            if label is not None and int(label) in selected:
                item.setSelected(True)

        if selected:
            for row in range(self._list.count()):
                item = self._list.item(row)
                if item is not None and item.isSelected():
                    self._list.setCurrentRow(row)
                    break
        else:
            self._list.setCurrentRow(-1)

        self._list.blockSignals(False)

    def set_active_label(self, label_id: int) -> None:
        self._select_label(label_id)

    def set_selection_state(
        self,
        label_ids: list[int],
        *,
        active_id: int | None = None,
    ) -> None:
        ids = set(label_ids)
        self._list.blockSignals(True)
        self._list.clearSelection()
        for row in range(self._list.count()):
            item = self._list.item(row)
            if item is None:
                continue
            label = item.data(Qt.UserRole)
            if label is not None and int(label) in ids:
                item.setSelected(True)
        if active_id is not None:
            self._select_label(active_id)
        elif not label_ids:
            self._list.setCurrentRow(-1)
        self._list.blockSignals(False)

    def _on_item_selection_changed(self) -> None:
        if self._updating:
            return
        label_ids: list[int] = []
        for item in self._list.selectedItems():
            label_id = item.data(Qt.UserRole)
            if label_id is not None:
                label_ids.append(int(label_id))
        self.labels_selected.emit(sorted(label_ids))

    def _select_label(self, label_id: int) -> None:
        self._list.blockSignals(True)
        for row in range(self._list.count()):
            item = self._list.item(row)
            if item is not None and item.data(Qt.UserRole) == label_id:
                self._list.setCurrentRow(row)
                break
        self._list.blockSignals(False)

    def _on_item_changed(self, item: QListWidgetItem) -> None:
        if self._updating:
            return
        label_id = item.data(Qt.UserRole)
        if label_id is None:
            return
        label_id = int(label_id)
        if item.checkState() == Qt.Checked:
            self._hidden_labels.discard(label_id)
        else:
            self._hidden_labels.add(label_id)
        self.label_visibility_changed.emit()

    def _on_current_item_changed(
        self,
        current: QListWidgetItem | None,
        _previous: QListWidgetItem | None,
    ) -> None:
        if current is None:
            return
        label_id = current.data(Qt.UserRole)
        if label_id is not None:
            self.label_id_changed.emit(int(label_id))


class SegmentationView(QMainWindow):
    """Main window UI for manual segmentation."""

    open_folder_requested = Signal()
    open_image_file_requested = Signal()
    save_requested = Signal()
    save_as_requested = Signal()
    undo_requested = Signal()
    redo_requested = Signal()
    tool_changed = Signal(str)
    label_id_changed = Signal(int)
    labels_selected = Signal(list)
    brush_size_changed = Signal(int)
    t_index_changed = Signal(int)
    z_index_changed = Signal(int)
    label_visibility_changed = Signal()
    channel_weights_changed = Signal(list)
    image_seg_blend_changed = Signal(int)
    closed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.resize(1280, 720)
        self.setWindowTitle("Cell-ACDC — Manual Segmentation")
        self._canvas = ImageCanvas()
        self._viewer = ViewerFrame(self._canvas)
        self._viewer.t_index_changed.connect(self.t_index_changed.emit)
        self._viewer.z_index_changed.connect(self.z_index_changed.emit)
        self._viewer.channel_weights_changed.connect(self.channel_weights_changed.emit)
        self._viewer.image_seg_blend_changed.connect(self.image_seg_blend_changed.emit)
        self._canvas.brush_size_changed.connect(self._on_canvas_brush_size_changed)
        self.setCentralWidget(self._viewer)
        self._build_actions()
        self._build_menu()
        self._build_options_bar()
        self._build_tool_rail()
        self._build_labels_dock()
        self.statusBar().showMessage("Open a Cell-ACDC folder to begin.")
        self._canvas.set_tool("hand")
        self._update_options_bar("hand")
        install_icon_theme_watcher(self, self._refresh_action_icons)

    def _refresh_action_icons(self) -> None:
        self._open_folder_act.setIcon(themed_lucide_qicon(LucideIcon.FOLDER_OPEN))
        self._open_file_act.setIcon(themed_lucide_qicon(LucideIcon.FILE_IMAGE))
        self._save_act.setIcon(themed_lucide_qicon(LucideIcon.SAVE))
        self._save_as_act.setIcon(themed_lucide_qicon(LucideIcon.SAVE_AS))
        self._undo_act.setIcon(themed_lucide_qicon(LucideIcon.UNDO))
        self._redo_act.setIcon(themed_lucide_qicon(LucideIcon.REDO))
        self._hand_act.setIcon(themed_lucide_qicon(LucideIcon.HAND))
        self._move_act.setIcon(themed_lucide_qicon(LucideIcon.MOVE))
        self._brush_act.setIcon(themed_lucide_qicon(LucideIcon.BRUSH))
        self._eraser_act.setIcon(themed_lucide_qicon(LucideIcon.ERASER))

    def _build_actions(self) -> None:
        self._open_folder_act = QAction("Open folder…", self)
        self._open_folder_act.setIcon(themed_lucide_qicon(LucideIcon.FOLDER_OPEN))
        self._open_folder_act.setShortcut("Ctrl+O")
        self._open_folder_act.setToolTip("Open Cell-ACDC experiment folder")
        self._open_folder_act.triggered.connect(self.open_folder_requested.emit)

        self._open_file_act = QAction("Open image file…", self)
        self._open_file_act.setIcon(themed_lucide_qicon(LucideIcon.FILE_IMAGE))
        self._open_file_act.setToolTip("Open a single image file")
        self._open_file_act.triggered.connect(self.open_image_file_requested.emit)

        self._save_act = QAction("Save mask", self)
        self._save_act.setIcon(themed_lucide_qicon(LucideIcon.SAVE))
        self._save_act.setShortcut("Ctrl+S")
        self._save_act.triggered.connect(self.save_requested.emit)

        self._save_as_act = QAction("Save mask as…", self)
        self._save_as_act.setIcon(themed_lucide_qicon(LucideIcon.SAVE_AS))
        self._save_as_act.setShortcut("Ctrl+Shift+S")
        self._save_as_act.triggered.connect(self.save_as_requested.emit)

        self._undo_act = QAction("Undo", self)
        self._undo_act.setIcon(themed_lucide_qicon(LucideIcon.UNDO))
        self._undo_act.setShortcut("Ctrl+Z")
        self._undo_act.triggered.connect(self.undo_requested.emit)

        self._redo_act = QAction("Redo", self)
        self._redo_act.setIcon(themed_lucide_qicon(LucideIcon.REDO))
        self._redo_act.setShortcut("Ctrl+Y")
        self._redo_act.triggered.connect(self.redo_requested.emit)

        self._hand_act = QAction("Hand", self)
        self._hand_act.setIcon(themed_lucide_qicon(LucideIcon.HAND))
        self._hand_act.setCheckable(True)
        self._hand_act.setShortcut("H")
        self._hand_act.setToolTip("Hand — pan the canvas (H)")
        self._hand_act.triggered.connect(lambda: self._on_tool_action("hand"))

        self._move_act = QAction("Move", self)
        self._move_act.setIcon(themed_lucide_qicon(LucideIcon.MOVE))
        self._move_act.setCheckable(True)
        self._move_act.setShortcut("V")
        self._move_act.setToolTip("Move — click or drag to select labels (V)")
        self._move_act.triggered.connect(lambda: self._on_tool_action("move"))

        self._brush_act = QAction("Brush", self)
        self._brush_act.setIcon(themed_lucide_qicon(LucideIcon.BRUSH))
        self._brush_act.setCheckable(True)
        self._brush_act.setShortcut("B")
        self._brush_act.setToolTip("Brush (B)")
        self._brush_act.triggered.connect(lambda: self._on_tool_action("brush"))

        self._eraser_act = QAction("Eraser", self)
        self._eraser_act.setIcon(themed_lucide_qicon(LucideIcon.ERASER))
        self._eraser_act.setCheckable(True)
        self._eraser_act.setShortcut("E")
        self._eraser_act.setToolTip("Eraser (E)")
        self._eraser_act.triggered.connect(lambda: self._on_tool_action("eraser"))

        self._hand_act.setChecked(True)

        self._tool_group = QActionGroup(self)
        self._tool_group.setExclusive(True)
        self._tool_group.addAction(self._move_act)
        self._tool_group.addAction(self._hand_act)
        self._tool_group.addAction(self._brush_act)
        self._tool_group.addAction(self._eraser_act)

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction(self._open_folder_act)
        file_menu.addAction(self._open_file_act)
        file_menu.addSeparator()
        file_menu.addAction(self._save_act)
        file_menu.addAction(self._save_as_act)
        file_menu.addSeparator()
        quit_act = QAction("&Quit", self)
        quit_act.setShortcut("Ctrl+Q")
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

        edit_menu = self.menuBar().addMenu("&Edit")
        edit_menu.addAction(self._undo_act)
        edit_menu.addAction(self._redo_act)

    def _build_options_bar(self) -> None:
        """Photoshop-style context options for the active tool."""
        bar = QToolBar("Options", self)
        bar.setMovable(False)
        self.addToolBar(bar)

        self._options_label = QLabel(" Label ID:")
        bar.addWidget(self._options_label)
        self._label_spin = QSpinBox()
        self._label_spin.setRange(1, 999_999)
        self._label_spin.setValue(1)
        self._label_spin.valueChanged.connect(self.label_id_changed.emit)
        bar.addWidget(self._label_spin)

        bar.addSeparator()
        self._size_label = QLabel(" Size:")
        bar.addWidget(self._size_label)
        self._size_spin = QSpinBox()
        self._size_spin.setRange(1, 128)
        self._size_spin.setValue(4)
        self._size_spin.valueChanged.connect(self.brush_size_changed.emit)
        bar.addWidget(self._size_spin)

    def _build_tool_rail(self) -> None:
        bar = QToolBar("Tools", self)
        bar.setOrientation(Qt.Vertical)
        bar.setMovable(False)
        bar.setToolButtonStyle(Qt.ToolButtonIconOnly)
        self.addToolBar(Qt.LeftToolBarArea, bar)
        bar.addAction(self._move_act)
        bar.addAction(self._hand_act)
        bar.addAction(self._brush_act)
        bar.addAction(self._eraser_act)

    def _build_labels_dock(self) -> None:
        self._label_panel = LabelListPanel()
        self._label_panel.label_id_changed.connect(self.label_id_changed.emit)
        self._label_panel.labels_selected.connect(self.labels_selected.emit)
        self._label_panel.label_visibility_changed.connect(
            self.label_visibility_changed.emit
        )

        dock = QDockWidget("Labels", self)
        dock.setWidget(self._label_panel)
        dock.setFeatures(
            QDockWidget.DockWidgetClosable
            | QDockWidget.DockWidgetMovable
            | QDockWidget.DockWidgetFloatable
        )
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

    def reset_label_visibility(self) -> None:
        self._label_panel.reset_visibility()

    def _on_canvas_brush_size_changed(self, size: int) -> None:
        self._size_spin.blockSignals(True)
        self._size_spin.setValue(size)
        self._size_spin.blockSignals(False)
        self.brush_size_changed.emit(size)

    def _on_tool_action(self, tool: str) -> None:
        self._canvas.set_tool(tool)
        self._update_options_bar(tool)
        self.tool_changed.emit(tool)

    def _update_options_bar(self, tool: str) -> None:
        is_brush = tool == "brush"
        is_paint = tool in {"brush", "eraser"}
        self._options_label.setVisible(is_brush)
        self._label_spin.setVisible(is_brush)
        self._size_label.setVisible(is_paint)
        self._size_spin.setVisible(is_paint)

    def ask_open_folder_path(self) -> str | None:
        path = QFileDialog.getExistingDirectory(
            self,
            "Open Cell-ACDC folder",
            "",
        )
        return path or None

    def ask_open_image_path(self) -> str | None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open image file",
            "",
            "Images (*.tif *.tiff *.npy *.npz);;All files (*)",
        )
        return path or None

    def ask_pick_position(self, names: list[str]) -> str | None:
        if not names:
            return None
        if len(names) == 1:
            return names[0]
        return self._pick_from_list("Select position", names)

    def ask_pick_channels(self, names: list[str]) -> list[str] | None:
        if not names:
            return None
        if len(names) == 1:
            return names
        return pick_many_from_list(self, "Select channels", names)

    def _pick_from_list(self, title: str, names: list[str]) -> str | None:
        return pick_from_list(self, title, names)

    def ask_save_mask_path(self) -> str | None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save mask",
            "",
            "NumPy zip (*.npz)",
        )
        return path or None

    def set_navigation(self, t: int, t_max: int, z: int, z_max: int) -> None:
        self._viewer.set_navigation(t, t_max, z, z_max)

    def update_navigation_indices(self, t: int, t_max: int, z: int, z_max: int) -> None:
        self._viewer.update_navigation_indices(t, t_max, z, z_max)

    def set_status(self, text: str) -> None:
        self.statusBar().showMessage(text)

    def set_paint_label_id(self, label_id: int) -> None:
        self._label_spin.blockSignals(True)
        self._label_spin.setValue(label_id)
        self._label_spin.blockSignals(False)

    def set_label_list(
        self,
        label_ids: list[int],
        *,
        selected_ids: list[int] | None = None,
    ) -> None:
        self._label_panel.set_label_list(label_ids, selected_ids=selected_ids)

    def set_label_selection(
        self,
        label_ids: list[int],
        *,
        active_id: int | None = None,
    ) -> None:
        self._label_panel.set_selection_state(label_ids, active_id=active_id)

    def set_selection_overlay(self, label_ids: list[int], mask_slice: np.ndarray) -> None:
        self._canvas.set_selection(label_ids, mask_slice)

    def get_hidden_label_ids(self) -> set[int]:
        return self._label_panel.get_hidden_label_ids()

    def set_brush_size(self, size: int) -> None:
        self._size_spin.blockSignals(True)
        self._size_spin.setValue(size)
        self._size_spin.blockSignals(False)
        self._canvas.set_brush_size(size)

    def refresh_display(
        self,
        channel_slices: list[np.ndarray],
        mask_slice: np.ndarray,
        *,
        display_levels: list[tuple[float, float] | None] | None = None,
        channel_labels: list[str] | None = None,
    ) -> None:
        self._canvas.set_channel_slices(
            channel_slices,
            display_levels=display_levels,
            labels=channel_labels,
        )
        self._canvas.set_mask_labels(mask_slice)
        self._canvas.set_hidden_labels(self.get_hidden_label_ids())

    def set_blend_ui(
        self,
        *,
        visible: bool,
        channel_names: list[str],
        channel_weights: list[float],
        image_seg: int,
    ) -> None:
        self._viewer.set_blend_controls(
            visible=visible,
            channel_names=channel_names,
            channel_weights=channel_weights,
            image_seg=image_seg,
        )

    def refresh_mask(self, mask_slice: np.ndarray) -> None:
        self._canvas.set_mask_labels(mask_slice)

    def refresh_label_visibility(self) -> None:
        self._canvas.set_hidden_labels(self.get_hidden_label_ids())

    @property
    def canvas(self) -> ImageCanvas:
        return self._canvas

    def closeEvent(self, event: QCloseEvent) -> None:
        self.closed.emit()
        super().closeEvent(event)
