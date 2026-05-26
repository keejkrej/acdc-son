"""Segmentation presenter: connects model and view."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from qtpy.QtWidgets import QMessageBox

from acdc.core.data import AcdcData, AcdcResult, load_segmentation
from acdc.utils.channels import channel_display_name

from acdc.core import experiment
from .model import SegmentationModel
from .editing import apply_label_visibility
from .view import SegmentationView


class SegmentationPresenter:
    """MVP presenter for manual segmentation."""

    def __init__(self, model: SegmentationModel, view: SegmentationView) -> None:
        self._model = model
        self._view = view
        self._selected_label_ids: list[int] = []
        self._connect()
        self._model.tool = "hand"

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
        v.labels_selected.connect(self._on_labels_selected)
        v.brush_size_changed.connect(self._on_brush_size_changed)
        v.t_index_changed.connect(self._on_t_changed)
        v.z_index_changed.connect(self._on_z_changed)
        v.label_visibility_changed.connect(self._on_label_visibility_changed)
        v.channel_weights_changed.connect(self._on_channel_weights_changed)
        v.image_seg_blend_changed.connect(self._on_image_seg_blend_changed)

        c = v.canvas
        c.paint_at.connect(self._on_paint_at)
        c.stroke_started.connect(self._model.begin_stroke)
        c.stroke_finished.connect(self._on_stroke_finished)
        c.pick_at.connect(self._on_pick_at)
        c.rect_pick.connect(self._on_rect_pick)

    def run(self) -> None:
        self._view.show()

    def open(
        self,
        images: Sequence[AcdcData],
        result: AcdcResult,
    ) -> None:
        """Load programmatic ``AcdcData`` channel(s) and ``AcdcResult``."""
        self._model.open(images, result)
        self._view.reset_label_visibility()
        self._selected_label_ids = []
        self._sync_controls()
        self._refresh()

    def _on_channel_weights_changed(self, weights: list[float]) -> None:
        self._model.set_channel_weights(weights)
        self._view.canvas.set_channel_weights(list(self._model.channel_weights))

    def _on_image_seg_blend_changed(self, value: int) -> None:
        self._model.set_image_seg_blend(value)
        self._view.canvas.set_image_seg_blend(value)

    def _sync_blend_ui(self) -> None:
        names = [
            channel_display_name(channel, index)
            for index, channel in enumerate(self._model.channels)
        ]
        self._view.set_blend_ui(
            visible=self._model.has_data,
            channel_names=names,
            channel_weights=self._model.channel_weights,
            image_seg=int(round(self._model.image_seg_blend)),
        )

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
            _basename, channel_names = experiment.discover_basename_and_channels(images_path)
        except Exception as exc:
            QMessageBox.critical(self._view, "Open failed", str(exc))
            return

        channels = self._view.ask_pick_channels(channel_names)
        if not channels:
            return

        try:
            images = AcdcData.from_experiment(images_path, channels=channels)
            mask_path = experiment.mask_path(images_path)
            result = load_segmentation(mask_path, like=images[0])
            self.open(images, result)
            self._model.mask_path = mask_path
        except Exception as exc:
            QMessageBox.critical(self._view, "Open failed", str(exc))
            return

    def _on_open_file(self) -> None:
        path = self._view.ask_open_image_path()
        if not path:
            return
        try:
            image_path = Path(path)
            imaged = AcdcData.from_path(image_path)
            mask_path = experiment.mask_path_for_image(image_path)
            result = load_segmentation(mask_path, like=imaged)
            self.open([imaged], result)
            self._model.mask_path = mask_path
        except Exception as exc:
            QMessageBox.critical(self._view, "Open failed", str(exc))
            return

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

    def _on_labels_selected(self, label_ids: list[int]) -> None:
        active_id = label_ids[0] if label_ids else self._model.label_id
        self._apply_selection(label_ids, active_id=active_id)

    def _on_brush_size_changed(self, size: int) -> None:
        self._model.brush_size = size

    def _on_t_changed(self, t: int) -> None:
        self._model.t_index = t
        self._refresh_slice()

    def _on_z_changed(self, z: int) -> None:
        self._model.z_index = z
        self._refresh_slice()

    def _on_label_visibility_changed(self) -> None:
        self._view.refresh_label_visibility()
        self._refresh_selection()

    def _on_stroke_finished(self) -> None:
        self._model.end_stroke()
        self._refresh_mask_only()
        self._sync_controls()

    def _on_paint_at(self, y: int, x: int) -> None:
        self._model.paint(y, x)
        self._refresh_mask_only()
        self._view.set_label_list(
            self._model.all_label_ids(),
            selected_ids=self._selected_label_ids,
        )

    def _on_pick_at(self, y: int, x: int) -> None:
        label_id = self._model.label_at(y, x)
        if label_id > 0:
            self._apply_selection([label_id], active_id=label_id)
        else:
            self._apply_selection([], active_id=self._model.label_id)

    def _on_rect_pick(self, y0: int, x0: int, y1: int, x1: int) -> None:
        label_ids = self._model.labels_in_rect(y0, x0, y1, x1)
        active_id = label_ids[0] if label_ids else self._model.label_id
        self._apply_selection(label_ids, active_id=active_id)

    def _apply_selection(self, label_ids: list[int], *, active_id: int) -> None:
        self._selected_label_ids = list(label_ids)
        if label_ids:
            self._model.label_id = active_id
            self._view.set_paint_label_id(active_id)
        self._view.set_label_selection(
            label_ids,
            active_id=active_id if label_ids else None,
        )
        self._refresh_selection()

    def _refresh_selection(self) -> None:
        if not self._model.has_data:
            return
        self._view.set_selection_overlay(
            self._selected_label_ids,
            self._model.current_mask_slice(),
        )

    def _sync_controls(self) -> None:
        stack_shape = self._model.stack_shape
        if stack_shape is None:
            return
        t_max = max(0, stack_shape.size_t - 1)
        z_max = max(0, stack_shape.size_z - 1)
        self._view.set_navigation(self._model.t_index, t_max, self._model.z_index, z_max)
        self._view.set_paint_label_id(self._model.label_id)
        self._view.set_brush_size(self._model.brush_size)
        self._view.set_label_list(self._model.all_label_ids())
        label = self._model.status_label()
        unsaved = " *" if not self._model.saved else ""
        self._view.set_status(f"{label}{unsaved}")

    def _display_mask(self):
        mask = self._model.current_mask_slice()
        return apply_label_visibility(mask, self._view.get_hidden_label_ids())

    def _refresh_slice(self) -> None:
        """Update the current T/Z slice only (frame slider hot path)."""
        if not self._model.has_data or self._model.stack_shape is None:
            return
        stack_shape = self._model.stack_shape
        t_max = max(0, stack_shape.size_t - 1)
        z_max = max(0, stack_shape.size_z - 1)
        labels = [
            channel_display_name(channel, index)
            for index, channel in enumerate(self._model.channels)
        ]
        self._view.refresh_display(
            self._model.current_channel_slices(),
            self._model.current_mask_slice(),
            display_levels=self._model.channel_display_levels,
            channel_labels=labels,
        )
        self._refresh_selection()
        self._view.update_navigation_indices(
            self._model.t_index,
            t_max,
            self._model.z_index,
            z_max,
        )

    def _refresh_view(self) -> None:
        if not self._model.has_data:
            return
        self._view.canvas.set_channel_weights(self._model.channel_weights)
        self._view.canvas.set_image_seg_blend(self._model.image_seg_blend)
        self._view.canvas.set_mask_max_label_id(self._model.max_label_id())
        self._refresh_slice()

    def _refresh(self) -> None:
        if not self._model.has_data:
            return
        self._view.canvas.set_channel_weights(self._model.channel_weights)
        self._view.canvas.set_image_seg_blend(self._model.image_seg_blend)
        self._view.canvas.set_mask_max_label_id(self._model.max_label_id())
        self._refresh_slice()
        self._sync_controls()
        self._sync_blend_ui()

    def _refresh_mask_only(self) -> None:
        if not self._model.has_data:
            return
        self._view.canvas.set_mask_max_label_id(self._model.max_label_id())
        self._view.refresh_mask(self._model.current_mask_slice())
        self._view.refresh_label_visibility()
        self._refresh_selection()
