"""Presenter for the 3D volume viewer."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from qtpy.QtWidgets import QMessageBox

from acdc.utils.channels import channel_display_name
from acdc.core.data import AcdcData, AcdcResult
from acdc.core import experiment
from acdc.volume.model import VolumeModel
from acdc.volume.prepare import (
    label_volume_for_vispy,
    mask_volume_zyx,
    normalize_image_stack_volume,
    volume_zyx,
)
from acdc.volume.view import VolumeView


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
        v.channel_weights_changed.connect(self._on_channel_weights_changed)
        v.image_seg_blend_changed.connect(self._on_image_seg_blend_changed)

    def run(self) -> None:
        self._view.show()

    def open(
        self,
        images: Sequence[AcdcData],
        result: AcdcResult | None = None,
        *,
        t_index: int = 0,
    ) -> AcdcResult:
        mask = self._model.bind(images, result)
        self._model.t_index = t_index
        self._view.reset_label_visibility()
        self._sync_controls()
        self._refresh()
        return mask

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
            self.open(images)
        except Exception as exc:
            QMessageBox.critical(self._view, "Open failed", str(exc))

    def _on_open_file(self) -> None:
        path = self._view.ask_open_image_path()
        if not path:
            return
        try:
            imaged = AcdcData.from_path(Path(path))
            self.open([imaged])
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
        if not self._model.has_data or self._model.primary is None:
            return
        stack_shape = self._model.primary.stack_shape
        t_max = max(0, stack_shape.size_t - 1)
        self._view.set_navigation(self._model.t_index, t_max, 0, 0)
        self._view.set_label_list(
            self._model.all_label_ids(),
            active_id=self._model.label_id,
        )
        self._view.set_status(self._model.status_label())

    def _refresh_view(self) -> None:
        if not self._model.has_data or self._model.primary is None or self._model.result is None:
            return
        reference = self._model.primary
        result = self._model.result
        volumes: list = []
        clims: list[tuple[float, float]] = []
        labels: list[str] = []
        for index, channel in enumerate(self._model.channels):
            image_raw = volume_zyx(channel, t_index=self._model.t_index)
            image_vol, image_clim = normalize_image_stack_volume(
                image_raw,
                channel.image,
                channel.stack_shape,
                stack_levels=self._model.channel_stack_levels[index],
                display_clim=self._model.channel_display_clim[index],
            )
            volumes.append(image_vol)
            clims.append(image_clim)
            labels.append(channel_display_name(channel, index))

        label_raw = mask_volume_zyx(result, reference.stack_shape, t_index=self._model.t_index)
        label_vol, lut_size = label_volume_for_vispy(label_raw)
        max_from_ids = max(self._model.all_label_ids(), default=0)
        lut_size = max(lut_size, max_from_ids + 1, 2)
        self._view.canvas.set_voxel_sizes(
            reference.physical_size_z,
            reference.physical_size_y,
            reference.physical_size_x,
        )
        self._view.canvas.set_image_channels(
            volumes,
            clims=clims,
            label_volume=label_vol,
            label_lut_size=lut_size,
            channel_labels=labels,
        )
        self._view.canvas.set_channel_weights(list(self._model.channel_weights))
        self._view.canvas.set_image_seg_blend(self._model.image_seg_blend)
        self._view.refresh_label_visibility()
        stack_shape = reference.stack_shape
        t_max = max(0, stack_shape.size_t - 1)
        self._view.update_navigation_indices(self._model.t_index, t_max, 0, 0)

    def _refresh(self) -> None:
        if not self._model.has_data:
            return
        title = self._model.status_label()
        if title:
            self._view.setWindowTitle(f"Cell-ACDC — 3D Volume — {title}")
        self._refresh_view()
        self._sync_controls()
        self._sync_blend_ui()
