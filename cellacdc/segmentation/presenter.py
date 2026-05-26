"""Segmentation presenter: connects model and view."""

from __future__ import annotations

from pathlib import Path

from qtpy.QtWidgets import QApplication, QMessageBox

from . import experiment
from .model import SegmentationModel
from .view import SegmentationView


class SegmentationPresenter:
    """MVP presenter for manual segmentation."""

    def __init__(self, model: SegmentationModel, view: SegmentationView) -> None:
        self._model = model
        self._view = view
        self._show_mask = True
        self._connect()

    def _connect(self) -> None:
        v = self._view
        v.open_folder_requested.connect(self._on_open_folder)
        v.open_image_file_requested.connect(self._on_open_file)
        v.save_requested.connect(self._on_save)
        v.save_as_requested.connect(self._on_save_as)
        v.undo_requested.connect(self._on_undo)
        v.redo_requested.connect(self._on_redo)
        v.tool_changed.connect(self._on_tool_changed)
        v.label_id_changed.connect(self._on_label_id_changed)
        v.brush_size_changed.connect(self._on_brush_size_changed)
        v.t_index_changed.connect(self._on_t_changed)
        v.z_index_changed.connect(self._on_z_changed)
        v.show_mask_toggled.connect(self._on_show_mask_toggled)

        c = v.canvas
        c.paint_at.connect(self._on_paint_at)
        c.stroke_started.connect(self._model.begin_stroke)
        c.stroke_finished.connect(self._on_stroke_finished)

    def run(self) -> None:
        self._view.show()

    def _on_open_folder(self) -> None:
        path = self._view.ask_open_folder_path()
        if not path:
            return
        try:
            images_paths = experiment.resolve_images_paths(Path(path))
        except Exception as exc:
            QMessageBox.critical(self._view, "Open failed", str(exc))
            return

        if len(images_paths) > 1:
            position_names = experiment.list_positions(Path(path))
            if not position_names:
                position_names = [
                    experiment.position_name_from_images_path(p) or p.parent.name
                    for p in images_paths
                ]
            picked = self._view.ask_pick_position(position_names)
            if not picked:
                return
            try:
                images_path = experiment.images_path_for_position(
                    Path(path), images_paths, picked
                )
            except ValueError as exc:
                QMessageBox.critical(self._view, "Open failed", str(exc))
                return
        else:
            images_path = images_paths[0]

        try:
            _basename, channels = experiment.discover_basename_and_channels(images_path)
        except Exception as exc:
            QMessageBox.critical(self._view, "Open failed", str(exc))
            return

        channel = self._view.ask_pick_channel(channels)
        if not channel:
            return

        try:
            spec = experiment.build_load_spec(images_path, channel)
            self._model.load_position(spec)
        except Exception as exc:
            QMessageBox.critical(self._view, "Open failed", str(exc))
            return

        self._sync_controls()
        self._refresh()

    def _on_open_file(self) -> None:
        path = self._view.ask_open_image_path()
        if not path:
            return
        try:
            self._model.load_image(Path(path))
        except Exception as exc:
            QMessageBox.critical(self._view, "Open failed", str(exc))
            return
        self._sync_controls()
        self._refresh()

    def _on_save(self) -> None:
        if not self._model.has_data:
            return
        try:
            dest = self._model.save_mask()
        except Exception as exc:
            QMessageBox.critical(self._view, "Save failed", str(exc))
            return
        self._view.set_status(f"Saved mask to {dest}")

    def _on_save_as(self) -> None:
        if not self._model.has_data:
            return
        path = self._view.ask_save_mask_path()
        if not path:
            return
        if not path.endswith(".npz"):
            path += ".npz"
        try:
            dest = self._model.save_mask(Path(path))
        except Exception as exc:
            QMessageBox.critical(self._view, "Save failed", str(exc))
            return
        self._view.set_status(f"Saved mask to {dest}")

    def _on_undo(self) -> None:
        if self._model.undo():
            self._refresh()

    def _on_redo(self) -> None:
        if self._model.redo():
            self._refresh()

    def _on_tool_changed(self, tool: str) -> None:
        self._model.tool = tool

    def _on_label_id_changed(self, label_id: int) -> None:
        self._model.label_id = label_id

    def _on_brush_size_changed(self, size: int) -> None:
        self._model.brush_size = size

    def _on_t_changed(self, t: int) -> None:
        self._model.t_index = t
        self._refresh()

    def _on_z_changed(self, z: int) -> None:
        self._model.z_index = z
        self._refresh()

    def _on_show_mask_toggled(self, checked: bool) -> None:
        self._show_mask = checked
        self._refresh()

    def _on_paint_at(self, y: int, x: int) -> None:
        self._model.paint(y, x)
        self._refresh_mask_only()

    def _on_stroke_finished(self) -> None:
        self._model.end_stroke()
        self._refresh_mask_only()
        self._sync_controls()

    def _sync_controls(self) -> None:
        layout = self._model.layout
        if layout is None:
            return
        t_max = max(0, layout.size_t - 1)
        z_max = max(0, layout.size_z - 1)
        self._view.set_navigation(self._model.t_index, t_max, self._model.z_index, z_max)
        self._view.set_label_id(self._model.label_id)
        self._view.set_brush_size(self._model.brush_size)
        label = self._model.status_label()
        dirty = " *" if self._model.dirty else ""
        self._view.set_status(f"{label}{dirty}")

    def _refresh(self) -> None:
        if not self._model.has_data:
            return
        img = self._model.current_image_slice()
        mask = self._model.current_mask_slice()
        self._view.refresh_display(img, mask, show_mask=self._show_mask)
        self._sync_controls()

    def _refresh_mask_only(self) -> None:
        if not self._model.has_data:
            return
        self._view.refresh_mask(
            self._model.current_mask_slice(),
            show_mask=self._show_mask,
        )


def create_app() -> tuple[QApplication, SegmentationPresenter]:
    import pyqtgraph as pg

    pg.setConfigOptions(imageAxisOrder="row-major")
    app = QApplication([])
    model = SegmentationModel()
    view = SegmentationView()
    presenter = SegmentationPresenter(model, view)
    return app, presenter
