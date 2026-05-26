"""Public data types for programmatic use (decoupled from the GUI)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from acdc.core import experiment, io, metadata, stack

if TYPE_CHECKING:
    import pandas as pd


@dataclass(frozen=True)
class AcdcData:
    """In-memory microscopy image volume for segmentation and 3D display."""

    image: np.ndarray
    stack_shape: stack.StackShape
    name: str = ""
    physical_size_z: float = 1.0
    physical_size_y: float = 1.0
    physical_size_x: float = 1.0

    @classmethod
    def from_experiment(
        cls,
        path: str | Path,
        *,
        channels: Sequence[str] | None = None,
        channel: str | None = None,
        position: str | None = None,
    ) -> tuple[AcdcData, ...]:
        """Load one or more channels from a Cell-ACDC experiment path."""
        path = Path(path)
        images_paths = experiment.resolve_images_paths(path)
        if len(images_paths) > 1:
            if position is None:
                names = experiment.list_positions(path)
                raise ValueError(
                    "Multiple positions found; pass position= "
                    f"({', '.join(names) or 'unknown'})"
                )
            images_path = experiment.images_path_for_position(
                path, images_paths, position
            )
        else:
            images_path = images_paths[0]

        _basename, available = experiment.discover_basename_and_channels(images_path)
        if channels is not None:
            if not channels:
                raise ValueError("At least one channel is required")
            chosen = list(channels)
        else:
            chosen = [channel or available[0]]
        for name in chosen:
            if name not in available:
                raise ValueError(
                    f'Channel "{name}" not found; available: {", ".join(available)}'
                )

        loaded = [cls._load_experiment_channel(images_path, name) for name in chosen]
        return coalesce_images(loaded)

    @classmethod
    def _load_experiment_channel(cls, images_path: Path, channel: str) -> AcdcData:
        image_path, meta, position_name = experiment.resolve_channel(images_path, channel)
        image = io.load_image(image_path)
        stack_shape = stack.shape_from_metadata(image.shape, meta.size_t, meta.size_z)
        if position_name and channel:
            name = f"{position_name} / {channel}"
        else:
            name = image_path.name
        return cls(
            image=image,
            stack_shape=stack_shape,
            name=name,
            physical_size_z=meta.physical_size_z,
            physical_size_y=meta.physical_size_y,
            physical_size_x=meta.physical_size_x,
        )

    @classmethod
    def from_path(cls, path: str | Path) -> AcdcData:
        """Load a single TIFF/NPY/NPZ file."""
        path = Path(path)
        image = io.load_image(path)
        meta = metadata.read_images_metadata(path.parent)
        stack_shape = stack.shape_from_metadata(image.shape, meta.size_t, meta.size_z)
        position_name = (
            experiment.position_name_from_images_path(path.parent)
            if path.parent.name == "Images"
            else None
        )
        channel_name = (
            experiment.channel_name_from_file(path.name, meta.basename)
            if meta.basename
            else None
        )
        if position_name and channel_name:
            name = f"{position_name} / {channel_name}"
        elif channel_name:
            name = f"{path.name} ({channel_name})"
        else:
            name = path.name
        return cls(
            image=image,
            stack_shape=stack_shape,
            name=name,
            physical_size_z=meta.physical_size_z,
            physical_size_y=meta.physical_size_y,
            physical_size_x=meta.physical_size_x,
        )

    @classmethod
    def from_arrays(
        cls,
        image: np.ndarray,
        *,
        size_t: int | None = None,
        size_z: int | None = None,
        name: str = "",
    ) -> AcdcData:
        """Wrap an in-memory array (no filesystem)."""
        image = np.asarray(image)
        stack_shape = stack.shape_from_metadata(image.shape, size_t, size_z)
        return cls(image=image, stack_shape=stack_shape, name=name or "array")


def coalesce_images(images: Sequence[AcdcData]) -> tuple[AcdcData, ...]:
    """Validate a channel list and return it as a tuple."""
    seq = tuple(images)
    if not seq:
        raise ValueError("At least one AcdcData is required")
    reference = seq[0]
    for other in seq[1:]:
        if other.image.shape != reference.image.shape:
            raise ValueError(
                f"Channel shape {other.image.shape} does not match "
                f"reference shape {reference.image.shape}"
            )
        if other.stack_shape != reference.stack_shape:
            raise ValueError("All channels must share the same stack shape")
    return seq


@dataclass
class AcdcResult:
    """In-memory label mask volume for segmentation and 3D overlay."""

    mask: np.ndarray
    mask_tracked: np.ndarray | None = None
    measurement: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        self.mask = np.asarray(self.mask, dtype=np.uint32)
        if self.mask_tracked is not None:
            self.mask_tracked = np.asarray(self.mask_tracked, dtype=np.uint32)

    @classmethod
    def empty_like(cls, imaged: AcdcData) -> AcdcResult:
        return cls(io.empty_mask_like(imaged.image))


def load_segmentation(path: str | Path, *, like: AcdcData) -> AcdcResult:
    """Load a mask from disk, or return an empty mask when the file is missing."""
    path = Path(path)
    if path.is_file():
        mask = io.load_mask(path)
        if mask.shape != like.image.shape:
            raise ValueError(
                f"Mask shape {mask.shape} does not match image shape {like.image.shape}"
            )
        return AcdcResult(mask)
    return AcdcResult.empty_like(like)


def save_segmentation(result: AcdcResult, path: str | Path) -> Path:
    """Persist a mask to NPZ (Cell-ACDC ``arr_0`` convention)."""
    dest = Path(path)
    io.save_mask(dest, result.mask)
    return dest


def _default_mask_path(
    folder: Path,
    *,
    position: str | None = None,
    channel: str | None = None,
    channels: Sequence[str] | None = None,
) -> Path:
    folder = Path(folder)
    if folder.is_file():
        return experiment.mask_path_for_image(folder)

    images_paths = experiment.resolve_images_paths(folder)
    if len(images_paths) > 1:
        if position is None:
            names = experiment.list_positions(folder)
            raise ValueError(
                "Multiple positions found; pass position= "
                f"({', '.join(names) or 'unknown'})"
            )
        images_path = experiment.images_path_for_position(folder, images_paths, position)
    else:
        images_path = images_paths[0]

    basename, available = experiment.discover_basename_and_channels(images_path)
    if channels is not None:
        chosen = channels[0]
    else:
        chosen = channel or available[0]
    return experiment.segm_file_path(images_path, basename)


def load(
    path: str | Path,
    *,
    channels: Sequence[str] | None = None,
    channel: str | None = None,
    position: str | None = None,
    segmentation: AcdcResult | None = None,
) -> tuple[tuple[AcdcData, ...], AcdcResult]:
    """Load channel(s) and a matching segmentation mask (existing or new empty)."""
    path = Path(path)
    if path.is_file():
        images = (AcdcData.from_path(path),)
    elif channels is not None:
        images = AcdcData.from_experiment(path, channels=channels, position=position)
    else:
        images = AcdcData.from_experiment(path, channel=channel, position=position)
    if segmentation is None:
        mask_path = _default_mask_path(
            path, position=position, channel=channel, channels=channels
        )
        segmentation = load_segmentation(mask_path, like=images[0])
    return images, segmentation
