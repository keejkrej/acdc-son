"""Volume viewer main window (segmentation-style app shell)."""

from __future__ import annotations

from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QAction, QActionGroup, QCloseEvent
from qtpy.QtWidgets import (
    QDialog,
    QDockWidget,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QSlider,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from acdc.ui.blend_controls import BlendControlBar
from acdc.ui.dialogs import pick_from_list, pick_many_from_list
from acdc.ui.icons import LucideIcon, install_icon_theme_watcher, themed_lucide_qicon
from acdc.segment.view import LabelListPanel
from acdc.volume.canvas import VolumeCanvas


class VolumeViewerFrame(QWidget):
    """Volume canvas with crossfade and frame controls (no Z-slice scrub)."""

    t_index_changed = Signal(int)
    channel_weights_changed = Signal(list)
    image_seg_blend_changed = Signal(int)

    def __init__(self, canvas: VolumeCanvas, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.canvas = canvas
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(canvas, stretch=1)

        bottom = QWidget()
        bottom_layout = QVBoxLayout(bottom)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(0)

        self._blend_bar = BlendControlBar()
        self._blend_bar.channel_weights_changed.connect(self.channel_weights_changed.emit)
        self._blend_bar.image_seg_changed.connect(self.image_seg_blend_changed.emit)
        self._blend_bar.channel_weights_changed.connect(self._on_channel_weights_changed)
        self._blend_bar.image_seg_changed.connect(self._on_image_seg_changed)
        bottom_layout.addWidget(self._blend_bar)

        self._frame_row = QWidget()
        frame_layout = QHBoxLayout(self._frame_row)
        frame_layout.setContentsMargins(8, 4, 8, 4)
        frame_layout.addWidget(QLabel("Frame:"))
        self._t_slider = QSlider(Qt.Horizontal)
        self._t_slider.valueChanged.connect(self.t_index_changed.emit)
        frame_layout.addWidget(self._t_slider, stretch=1)
        self._t_label = QLabel("0/0")
        self._t_label.setMinimumWidth(48)
        frame_layout.addWidget(self._t_label)
        bottom_layout.addWidget(self._frame_row)
        self._frame_row.setVisible(False)

        layout.addWidget(bottom)
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
        del z, z_max
        show_frame = t_max > 0
        self._frame_row.setVisible(show_frame)
        self._t_slider.blockSignals(True)
        self._t_slider.setEnabled(show_frame)
        self._t_slider.setMinimum(0)
        self._t_slider.setMaximum(max(0, t_max))
        self._t_slider.setValue(t)
        self._t_slider.blockSignals(False)
        self._t_label.setText(f"{t}/{t_max}")

    def update_navigation_indices(self, t: int, t_max: int, z: int, z_max: int) -> None:
        del z, z_max
        self._t_label.setText(f"{t}/{t_max}")


class VolumeView(QMainWindow):
    """Main window for 3D volume viewing."""

    open_folder_requested = Signal()
    open_image_file_requested = Signal()
    label_id_changed = Signal(int)
    label_visibility_changed = Signal()
    t_index_changed = Signal(int)
    channel_weights_changed = Signal(list)
    image_seg_blend_changed = Signal(int)
    closed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.resize(1280, 720)
        self.setWindowTitle("Cell-ACDC — 3D Volume")
        self._canvas = VolumeCanvas()
        self._viewer = VolumeViewerFrame(self._canvas)
        self._viewer.t_index_changed.connect(self.t_index_changed.emit)
        self._viewer.channel_weights_changed.connect(self.channel_weights_changed.emit)
        self._viewer.image_seg_blend_changed.connect(self.image_seg_blend_changed.emit)
        self.setCentralWidget(self._viewer)
        self._build_actions()
        self._build_menu()
        self._build_tool_rail()
        self._build_labels_dock()
        self.statusBar().showMessage("Open a Cell-ACDC folder or image file to begin.")
        self._hand_act.setChecked(True)
        install_icon_theme_watcher(self, self._refresh_action_icons)

    def _refresh_action_icons(self) -> None:
        self._open_folder_act.setIcon(themed_lucide_qicon(LucideIcon.FOLDER_OPEN))
        self._open_file_act.setIcon(themed_lucide_qicon(LucideIcon.FILE_IMAGE))
        self._hand_act.setIcon(themed_lucide_qicon(LucideIcon.HAND))

    def _build_actions(self) -> None:
        self._open_folder_act = QAction("Open folder…", self)
        self._open_folder_act.setIcon(themed_lucide_qicon(LucideIcon.FOLDER_OPEN))
        self._open_folder_act.setShortcut("Ctrl+O")
        self._open_folder_act.triggered.connect(self.open_folder_requested.emit)

        self._open_file_act = QAction("Open image file…", self)
        self._open_file_act.setIcon(themed_lucide_qicon(LucideIcon.FILE_IMAGE))
        self._open_file_act.triggered.connect(self.open_image_file_requested.emit)

        self._hand_act = QAction("Hand", self)
        self._hand_act.setIcon(themed_lucide_qicon(LucideIcon.HAND))
        self._hand_act.setCheckable(True)
        self._hand_act.setShortcut("H")
        self._hand_act.setToolTip("Hand — navigate the 3D view (H)")
        self._hand_act.triggered.connect(self._on_hand_tool)

        self._tool_group = QActionGroup(self)
        self._tool_group.setExclusive(True)
        self._tool_group.addAction(self._hand_act)

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction(self._open_folder_act)
        file_menu.addAction(self._open_file_act)
        file_menu.addSeparator()
        quit_act = QAction("&Quit", self)
        quit_act.setShortcut("Ctrl+Q")
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

    def _build_tool_rail(self) -> None:
        bar = QToolBar("Tools", self)
        bar.setOrientation(Qt.Vertical)
        bar.setMovable(False)
        bar.setToolButtonStyle(Qt.ToolButtonIconOnly)
        self.addToolBar(Qt.LeftToolBarArea, bar)
        bar.addAction(self._hand_act)

    def _build_labels_dock(self) -> None:
        self._label_panel = LabelListPanel()
        self._label_panel.label_id_changed.connect(self.label_id_changed.emit)
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

    def _on_hand_tool(self) -> None:
        self._canvas.focus_canvas()

    def reset_label_visibility(self) -> None:
        self._label_panel.reset_visibility()

    def get_hidden_label_ids(self) -> set[int]:
        return self._label_panel.get_hidden_label_ids()

    def set_label_list(self, label_ids: list[int], *, active_id: int) -> None:
        self._label_panel.set_label_list(label_ids)
        self._label_panel.set_active_label(active_id)

    def set_active_label(self, label_id: int) -> None:
        self._label_panel.set_active_label(label_id)

    def set_navigation(self, t: int, t_max: int, z: int, z_max: int) -> None:
        self._viewer.set_navigation(t, t_max, z, z_max)

    def update_navigation_indices(self, t: int, t_max: int, z: int, z_max: int) -> None:
        self._viewer.update_navigation_indices(t, t_max, z, z_max)

    def refresh_label_visibility(self) -> None:
        self._canvas.set_hidden_labels(self.get_hidden_label_ids())

    def set_status(self, text: str) -> None:
        self.statusBar().showMessage(text)

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

    def ask_open_folder_path(self) -> str | None:
        from qtpy.QtWidgets import QFileDialog

        path = QFileDialog.getExistingDirectory(self, "Open Cell-ACDC folder", "")
        return path or None

    def ask_open_image_path(self) -> str | None:
        from qtpy.QtWidgets import QFileDialog

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

    @property
    def canvas(self) -> VolumeCanvas:
        return self._canvas

    def closeEvent(self, event: QCloseEvent) -> None:
        self.closed.emit()
        super().closeEvent(event)
