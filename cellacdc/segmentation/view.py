"""Segmentation view: Qt widgets and pyqtgraph display (no business logic)."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QActionGroup
from qtpy.QtWidgets import (
    QAction,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
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

from . import tools


class ImageCanvas(QWidget):
    """Pyqtgraph canvas that emits image coordinates on paint gestures."""

    paint_at = Signal(int, int)  # y, x
    stroke_started = Signal()
    stroke_finished = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._layout = pg.GraphicsLayoutWidget()
        layout.addWidget(self._layout)
        self._plot = self._layout.addPlot()
        self._plot.invertY(True)
        self._plot.setAspectLocked(True)
        self._plot.hideAxis("bottom")
        self._plot.hideAxis("left")
        self._image_item = pg.ImageItem()
        self._mask_item = pg.ImageItem()
        self._plot.addItem(self._image_item)
        self._plot.addItem(self._mask_item)
        self._mask_item.setZValue(10)
        self._label_lut = tools.build_label_lut()
        self._mask_item.setLookupTable(self._label_lut)
        self._mask_item.setLevels([0, len(self._label_lut)])
        self._painting = False
        self._viewbox = self._plot.getViewBox()
        self._orig_drag = self._viewbox.mouseDragEvent
        self._viewbox.mouseDragEvent = self._mouse_drag_event  # type: ignore[method-assign]

    def set_image(self, gray: np.ndarray) -> None:
        if self._image_item.image is None or self._image_item.image.shape != gray.shape:
            self._image_item.setImage(gray, autoLevels=True)
        else:
            self._image_item.setImage(gray, autoLevels=False)

    def set_mask_labels(self, labels: np.ndarray) -> None:
        max_label = int(labels.max(initial=0))
        if max_label >= self._label_lut.shape[0]:
            self._label_lut = tools.build_label_lut(max_label + 256)
            self._mask_item.setLookupTable(self._label_lut)
            self._mask_item.setLevels([0, len(self._label_lut)])
        current = self._mask_item.image
        if (
            current is not None
            and current.shape == labels.shape
            and np.may_share_memory(current, labels)
        ):
            self._mask_item._renderRequired = True
            self._mask_item.update()
            return
        self._mask_item.setImage(
            labels,
            autoLevels=False,
            levels=(0, len(self._label_lut) - 1),
        )

    def clear(self) -> None:
        self._image_item.clear()
        self._mask_item.clear()

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

    def _mouse_drag_event(self, ev, axis=None) -> None:
        if ev.button() != Qt.LeftButton:
            self._orig_drag(ev, axis)
            return
        if ev.isStart():
            self.stroke_started.emit()
            self._painting = True
        if self._painting:
            mapped = self._map_pos_to_pixel(ev.scenePos())
            if mapped is not None:
                self.paint_at.emit(*mapped)
        if ev.isFinish() and self._painting:
            self.stroke_finished.emit()
            self._painting = False
        ev.accept()


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
    """Canvas with bottom transport controls."""

    t_index_changed = Signal(int)
    z_index_changed = Signal(int)

    def __init__(self, canvas: ImageCanvas, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.canvas = canvas
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(canvas, stretch=1)
        self._transport = TransportBar()
        self._transport.t_index_changed.connect(self.t_index_changed.emit)
        self._transport.z_index_changed.connect(self.z_index_changed.emit)
        layout.addWidget(self._transport)

    def set_navigation(self, t: int, t_max: int, z: int, z_max: int) -> None:
        self._transport.set_navigation(t, t_max, z, z_max)

    def update_navigation_indices(self, t: int, t_max: int, z: int, z_max: int) -> None:
        self._transport.update_indices(t, t_max, z, z_max)


class LabelHeaderCheckBox(QCheckBox):
    """Explorer-style header: partial/unchecked -> check all, checked -> clear all."""

    def nextCheckState(self) -> None:
        if self.checkState() == Qt.Checked:
            self.setCheckState(Qt.Unchecked)
        else:
            self.setCheckState(Qt.Checked)


class LabelListPanel(QWidget):
    """Explorer-style label list with global and per-label visibility toggles."""

    label_id_changed = Signal(int)
    label_visibility_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._updating = False
        self._hidden_labels: set[int] = set()
        self._all_label_ids: list[int] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(8, 6, 8, 6)
        self._header_cb = LabelHeaderCheckBox("Labels")
        self._header_cb.setTristate(True)
        self._header_cb.setChecked(True)
        self._header_cb.checkStateChanged.connect(self._on_header_state_changed)
        header_layout.addWidget(self._header_cb)
        header_layout.addStretch()
        layout.addWidget(header)

        self._list = QListWidget()
        self._list.itemChanged.connect(self._on_item_changed)
        self._list.currentItemChanged.connect(self._on_current_item_changed)
        layout.addWidget(self._list)

    def reset_visibility(self) -> None:
        self._hidden_labels.clear()

    def get_hidden_label_ids(self) -> set[int]:
        return set(self._hidden_labels)

    def set_label_list(self, label_ids: list[int], *, active_id: int) -> None:
        self._all_label_ids = list(label_ids)

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

        self._sync_header_checkbox()
        self._select_label(active_id)

    def set_active_label(self, label_id: int) -> None:
        self._select_label(label_id)

    def _select_label(self, label_id: int) -> None:
        self._list.blockSignals(True)
        for row in range(self._list.count()):
            item = self._list.item(row)
            if item is not None and item.data(Qt.UserRole) == label_id:
                self._list.setCurrentRow(row)
                break
        self._list.blockSignals(False)

    def _on_header_state_changed(self, state: int) -> None:
        if self._updating or state == Qt.PartiallyChecked:
            return
        self._updating = True
        if state == Qt.Checked:
            self._hidden_labels.clear()
            check = Qt.Checked
        else:
            self._hidden_labels = set(self._all_label_ids)
            check = Qt.Unchecked
        for row in range(self._list.count()):
            item = self._list.item(row)
            if item is not None:
                item.setCheckState(check)
        self._updating = False
        self.label_visibility_changed.emit()

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
        self._sync_header_checkbox()
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

    def _sync_header_checkbox(self) -> None:
        total = len(self._all_label_ids)
        if total == 0:
            self._updating = True
            self._header_cb.setCheckState(Qt.Unchecked)
            self._updating = False
            return

        visible = sum(1 for label_id in self._all_label_ids if label_id not in self._hidden_labels)
        self._updating = True
        if visible == 0:
            self._header_cb.setCheckState(Qt.Unchecked)
        elif visible == total:
            self._header_cb.setCheckState(Qt.Checked)
        else:
            self._header_cb.setCheckState(Qt.PartiallyChecked)
        self._updating = False


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
    brush_size_changed = Signal(int)
    t_index_changed = Signal(int)
    z_index_changed = Signal(int)
    label_visibility_changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Cell-ACDC — Manual Segmentation")
        self._canvas = ImageCanvas()
        self._viewer = ViewerFrame(self._canvas)
        self._viewer.t_index_changed.connect(self.t_index_changed.emit)
        self._viewer.z_index_changed.connect(self.z_index_changed.emit)
        self.setCentralWidget(self._viewer)
        self._build_actions()
        self._build_menu()
        self._build_options_bar()
        self._build_tool_rail()
        self._build_labels_dock()
        self.statusBar().showMessage("Open a Cell-ACDC folder to begin.")
        self._update_options_bar("brush")

    def _build_actions(self) -> None:
        self._open_folder_act = QAction("Open folder…", self)
        self._open_folder_act.setShortcut("Ctrl+O")
        self._open_folder_act.setToolTip("Open Cell-ACDC experiment folder")
        self._open_folder_act.triggered.connect(self.open_folder_requested.emit)

        self._open_file_act = QAction("Open image file…", self)
        self._open_file_act.setToolTip("Open a single image file")
        self._open_file_act.triggered.connect(self.open_image_file_requested.emit)

        self._save_act = QAction("Save mask", self)
        self._save_act.setShortcut("Ctrl+S")
        self._save_act.triggered.connect(self.save_requested.emit)

        self._save_as_act = QAction("Save mask as…", self)
        self._save_as_act.setShortcut("Ctrl+Shift+S")
        self._save_as_act.triggered.connect(self.save_as_requested.emit)

        self._undo_act = QAction("Undo", self)
        self._undo_act.setShortcut("Ctrl+Z")
        self._undo_act.triggered.connect(self.undo_requested.emit)

        self._redo_act = QAction("Redo", self)
        self._redo_act.setShortcut("Ctrl+Y")
        self._redo_act.triggered.connect(self.redo_requested.emit)

        self._brush_act = QAction("Brush", self)
        self._brush_act.setCheckable(True)
        self._brush_act.setChecked(True)
        self._brush_act.setShortcut("B")
        self._brush_act.setToolTip("Brush (B)")
        self._brush_act.triggered.connect(lambda: self._on_tool_action("brush"))

        self._eraser_act = QAction("Eraser", self)
        self._eraser_act.setCheckable(True)
        self._eraser_act.setShortcut("X")
        self._eraser_act.setToolTip("Eraser (X)")
        self._eraser_act.triggered.connect(lambda: self._on_tool_action("eraser"))

        self._tool_group = QActionGroup(self)
        self._tool_group.setExclusive(True)
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
        bar.addWidget(QLabel(" Size:"))
        self._size_spin = QSpinBox()
        self._size_spin.setRange(1, 128)
        self._size_spin.setValue(4)
        self._size_spin.valueChanged.connect(self.brush_size_changed.emit)
        bar.addWidget(self._size_spin)

    def _build_tool_rail(self) -> None:
        bar = QToolBar("Tools", self)
        bar.setOrientation(Qt.Vertical)
        bar.setMovable(False)
        self.addToolBar(Qt.LeftToolBarArea, bar)
        bar.addAction(self._brush_act)
        bar.addAction(self._eraser_act)

    def _build_labels_dock(self) -> None:
        self._label_panel = LabelListPanel()
        self._label_panel.label_id_changed.connect(self.label_id_changed.emit)
        self._label_panel.label_visibility_changed.connect(
            self.label_visibility_changed.emit
        )

        dock = QDockWidget("Labels", self)
        dock.setWidget(self._label_panel)
        dock.setFeatures(
            QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetMovable
        )
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

    def reset_label_visibility(self) -> None:
        self._label_panel.reset_visibility()

    def _on_tool_action(self, tool: str) -> None:
        self._update_options_bar(tool)
        self.tool_changed.emit(tool)

    def _update_options_bar(self, tool: str) -> None:
        is_brush = tool == "brush"
        self._options_label.setVisible(is_brush)
        self._label_spin.setVisible(is_brush)

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

    def ask_pick_channel(self, names: list[str]) -> str | None:
        if not names:
            return None
        if len(names) == 1:
            return names[0]
        return self._pick_from_list("Select channel", names)

    def _pick_from_list(self, title: str, names: list[str]) -> str | None:
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel(f"{title}:"))
        list_widget = QListWidget()
        list_widget.addItems(names)
        list_widget.setCurrentRow(0)
        layout.addWidget(list_widget)
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            parent=dialog,
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() != QDialog.Accepted:
            return None
        selected = list_widget.currentItem()
        return selected.text() if selected is not None else None

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

    def set_label_id(self, label_id: int) -> None:
        self._label_spin.blockSignals(True)
        self._label_spin.setValue(label_id)
        self._label_spin.blockSignals(False)
        self._label_panel.set_active_label(label_id)

    def set_label_list(self, label_ids: list[int], *, active_id: int) -> None:
        self._label_panel.set_label_list(label_ids, active_id=active_id)

    def get_hidden_label_ids(self) -> set[int]:
        return self._label_panel.get_hidden_label_ids()

    def set_brush_size(self, size: int) -> None:
        self._size_spin.blockSignals(True)
        self._size_spin.setValue(size)
        self._size_spin.blockSignals(False)

    def refresh_display(
        self,
        image_slice: np.ndarray,
        mask_slice: np.ndarray,
    ) -> None:
        self._canvas.set_image(image_slice)
        self._canvas.set_mask_labels(mask_slice)

    def refresh_mask(self, mask_slice: np.ndarray) -> None:
        self._canvas.set_mask_labels(mask_slice)

    @property
    def canvas(self) -> ImageCanvas:
        return self._canvas
