"""Segmentation view: Qt widgets and pyqtgraph display (no business logic)."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QAction,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QListWidget,
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
    show_mask_toggled = Signal(bool)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Cell-ACDC — Manual Segmentation")
        self._canvas = ImageCanvas()
        self.setCentralWidget(self._build_central())
        self._build_menu()
        self._build_toolbar()
        self._build_nav_bar()

    def _build_central(self) -> QWidget:
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.addWidget(self._canvas)
        self._status = QLabel("Open a Cell-ACDC folder to begin.")
        layout.addWidget(self._status)
        return root

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        open_folder_act = QAction("Open &folder…", self)
        open_folder_act.setShortcut("Ctrl+O")
        open_folder_act.triggered.connect(self.open_folder_requested.emit)
        file_menu.addAction(open_folder_act)
        open_file_act = QAction("Open image &file…", self)
        open_file_act.triggered.connect(self.open_image_file_requested.emit)
        file_menu.addAction(open_file_act)
        save_act = QAction("&Save mask", self)
        save_act.setShortcut("Ctrl+S")
        save_act.triggered.connect(self.save_requested.emit)
        file_menu.addAction(save_act)
        save_as_act = QAction("Save mask &as…", self)
        save_as_act.setShortcut("Ctrl+Shift+S")
        save_as_act.triggered.connect(self.save_as_requested.emit)
        file_menu.addAction(save_as_act)
        file_menu.addSeparator()
        quit_act = QAction("&Quit", self)
        quit_act.setShortcut("Ctrl+Q")
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

        edit_menu = self.menuBar().addMenu("&Edit")
        undo_act = QAction("&Undo", self)
        undo_act.setShortcut("Ctrl+Z")
        undo_act.triggered.connect(self.undo_requested.emit)
        edit_menu.addAction(undo_act)
        redo_act = QAction("&Redo", self)
        redo_act.setShortcut("Ctrl+Y")
        redo_act.triggered.connect(self.redo_requested.emit)
        edit_menu.addAction(redo_act)

    def _build_toolbar(self) -> None:
        bar = QToolBar("Tools", self)
        self.addToolBar(bar)
        self._brush_btn = bar.addAction("Brush")
        self._brush_btn.setCheckable(True)
        self._brush_btn.setChecked(True)
        self._brush_btn.setShortcut("B")
        self._brush_btn.triggered.connect(lambda: self._select_tool("brush"))

        self._eraser_btn = bar.addAction("Eraser")
        self._eraser_btn.setCheckable(True)
        self._eraser_btn.setShortcut("X")
        self._eraser_btn.triggered.connect(lambda: self._select_tool("eraser"))

        bar.addSeparator()
        bar.addWidget(QLabel(" Label ID: "))
        self._label_spin = QSpinBox()
        self._label_spin.setRange(1, 999_999)
        self._label_spin.setValue(1)
        self._label_spin.valueChanged.connect(self.label_id_changed.emit)
        bar.addWidget(self._label_spin)

        bar.addWidget(QLabel(" Brush size: "))
        self._size_spin = QSpinBox()
        self._size_spin.setRange(1, 128)
        self._size_spin.setValue(4)
        self._size_spin.valueChanged.connect(self.brush_size_changed.emit)
        bar.addWidget(self._size_spin)

        bar.addSeparator()
        self._show_mask = QCheckBox("Show mask")
        self._show_mask.setChecked(True)
        self._show_mask.toggled.connect(self.show_mask_toggled.emit)
        bar.addWidget(self._show_mask)

    def _build_nav_bar(self) -> None:
        bar = QToolBar("Navigate", self)
        self.addToolBar(bar)
        bar.addWidget(QLabel(" Frame: "))
        self._t_slider = QSlider(Qt.Horizontal)
        self._t_slider.valueChanged.connect(self.t_index_changed.emit)
        bar.addWidget(self._t_slider)
        self._t_label = QLabel("0/0")
        bar.addWidget(self._t_label)

        bar.addWidget(QLabel("  Z-slice: "))
        self._z_slider = QSlider(Qt.Horizontal)
        self._z_slider.valueChanged.connect(self.z_index_changed.emit)
        bar.addWidget(self._z_slider)
        self._z_label = QLabel("0/0")
        bar.addWidget(self._z_label)

    def _select_tool(self, tool: str) -> None:
        self._brush_btn.setChecked(tool == "brush")
        self._eraser_btn.setChecked(tool == "eraser")
        self.tool_changed.emit(tool)

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
        self._t_slider.blockSignals(True)
        self._z_slider.blockSignals(True)
        self._t_slider.setEnabled(t_max > 0)
        self._z_slider.setEnabled(z_max > 0)
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

    def set_status(self, text: str) -> None:
        self._status.setText(text)

    def set_label_id(self, label_id: int) -> None:
        self._label_spin.blockSignals(True)
        self._label_spin.setValue(label_id)
        self._label_spin.blockSignals(False)

    def set_brush_size(self, size: int) -> None:
        self._size_spin.blockSignals(True)
        self._size_spin.setValue(size)
        self._size_spin.blockSignals(False)

    def refresh_display(
        self,
        image_slice: np.ndarray,
        mask_slice: np.ndarray,
        *,
        show_mask: bool,
    ) -> None:
        self._canvas.set_image(image_slice)
        if show_mask:
            self._canvas.set_mask_labels(mask_slice)
        else:
            self._canvas.set_mask_labels(np.zeros_like(mask_slice))

    def refresh_mask(self, mask_slice: np.ndarray, *, show_mask: bool) -> None:
        if show_mask:
            self._canvas.set_mask_labels(mask_slice)
        else:
            self._canvas.set_mask_labels(np.zeros_like(mask_slice))

    @property
    def canvas(self) -> ImageCanvas:
        return self._canvas
