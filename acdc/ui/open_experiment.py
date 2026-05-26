"""Shared Cell-ACDC open-folder dialog flow."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from qtpy.QtWidgets import QMessageBox, QWidget

from acdc.core import experiment
from acdc.core.data import AcdcData


class ExperimentOpenView(Protocol):
    def ask_open_folder_path(self) -> str | None: ...

    def ask_pick_position(self, position_names: list[str]) -> str | None: ...

    def ask_pick_channels(self, channel_names: list[str]) -> list[str] | None: ...


def pick_experiment_images(
    parent: QWidget,
    view: ExperimentOpenView,
) -> tuple[Path, tuple[AcdcData, ...]] | None:
    """Run folder → position → channel pickers and load experiment images."""
    path = view.ask_open_folder_path()
    if not path:
        return None
    try:
        images_paths = experiment.resolve_images_paths(Path(path))
    except Exception as exc:
        QMessageBox.critical(parent, "Open failed", str(exc))
        return None

    if len(images_paths) > 1:
        position_names = experiment.list_positions(Path(path))
        if not position_names:
            position_names = [
                experiment.position_name_from_images_path(p) or p.parent.name
                for p in images_paths
            ]
        picked = view.ask_pick_position(position_names)
        if not picked:
            return None
        try:
            images_path = experiment.images_path_for_position(
                Path(path), images_paths, picked
            )
        except ValueError as exc:
            QMessageBox.critical(parent, "Open failed", str(exc))
            return None
    else:
        images_path = images_paths[0]

    try:
        _basename, channel_names = experiment.discover_basename_and_channels(images_path)
    except Exception as exc:
        QMessageBox.critical(parent, "Open failed", str(exc))
        return None

    channels = view.ask_pick_channels(channel_names)
    if not channels:
        return None

    try:
        images = AcdcData.from_experiment(images_path, channels=channels)
    except Exception as exc:
        QMessageBox.critical(parent, "Open failed", str(exc))
        return None
    return images_path, images
