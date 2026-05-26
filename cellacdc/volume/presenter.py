"""Presenter for the 3D volume viewer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from qtpy.QtWidgets import QMessageBox

from cellacdc.data import ImagedData, SegmentationResult
from cellacdc.segmentation import experiment
from cellacdc.volume.model import VolumeModel
from cellacdc.volume.prepare import (
    label_volume_for_vispy,
    mask_volume_zyx,
    normalize_image_volume,
    volume_zyx,
)
from cellacdc.volume.view import VolumeView


class VolumePresenter:
    """Connects ``VolumeModel`` to ``VolumeView``."""

    def __init__(self, model: VolumeModel, view: VolumeView) -> None:
        self._model = model
        self._view = view
        self._connect()

    def _connect(self) -> None:
        v = self._view
        v.open_folder_requested.connect(self._on_open_folder)
        v.open_image_file_requested.connect(self._on_open_file)
        v.label_id_changed.connect(self._on_label_id_changed)
        v.label_visibility_changed.connect(self._on_label_visibility_changed)
        v.t_index_changed.connect(self._on_t_changed)

    def run(self) -> None:
        self._view.show()

    def open(
        self,
        imaged: ImagedData,
        result: SegmentationResult | None = None,
        *,
        t_index: int = 0,
    ) -> SegmentationResult:
        mask = self._model.bind(imaged, result)
        self._model.t_index = t_index
        self._view.reset_label_visibility()
        self._sync_controls()
        self._refresh()
        return mask

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
            imaged = ImagedData.from_path(images_path, channel=channel)
            self.open(imaged)
        except Exception as exc:
            QMessageBox.critical(self._view, "Open failed", str(exc))

    def _on_open_file(self) -> None:
        path = self._view.ask_open_image_path()
        if not path:
            return
        try:
            imaged = ImagedData.from_image_path(Path(path))
            self.open(imaged)
        except Exception as exc:
            QMessageBox.critical(self._view, "Open failed", str(exc))

    def _on_label_id_changed(self, label_id: int) -> None:
        self._model.label_id = label_id

    def _on_label_visibility_changed(self) -> None:
        self._view.refresh_label_visibility()

    def _on_t_changed(self, t: int) -> None:
        self._model.t_index = t
        self._refresh_view()

    def _sync_controls(self) -> None:
        if not self._model.has_data or self._model.imaged is None:
            return
        layout = self._model.imaged.layout
        t_max = max(0, layout.size_t - 1)
        self._view.set_navigation(self._model.t_index, t_max, 0, 0)
        self._view.set_label_list(
            self._model.all_label_ids(),
            active_id=self._model.label_id,
        )
        self._view.set_status(self._model.status_label())

    def _refresh_volume(self) -> None:
        if not self._model.has_data or self._model.imaged is None or self._model.result is None:
            return
        imaged = self._model.imaged
        result = self._model.result
        image_raw = volume_zyx(imaged, t_index=self._model.t_index)
        image_vol, _clim = normalize_image_volume(image_raw)
        label_raw = mask_volume_zyx(result, imaged.layout, t_index=self._model.t_index)
        label_vol, lut_size = label_volume_for_vispy(label_raw)
        max_from_ids = max(self._model.all_label_ids(), default=0)
        lut_size = max(lut_size, max_from_ids + 1, 2)
        self._view.canvas.set_voxel_sizes(
            imaged.physical_size_z,
            imaged.physical_size_y,
            imaged.physical_size_x,
        )
        self._view.canvas.set_volumes(
            image_vol,
            label_vol,
            label_lut_size=lut_size,
        )
        self._view.refresh_label_visibility()

    def _refresh_view(self) -> None:
        if not self._model.has_data or self._model.imaged is None:
            return
        layout = self._model.imaged.layout
        t_max = max(0, layout.size_t - 1)
        self._refresh_volume()
        self._view.update_navigation_indices(self._model.t_index, t_max, 0, 0)

    def _refresh(self) -> None:
        if not self._model.has_data:
            return
        title = self._model.status_label()
        if title:
            self._view.setWindowTitle(f"Cell-ACDC — 3D Volume — {title}")
        self._refresh_view()
        self._sync_controls()
