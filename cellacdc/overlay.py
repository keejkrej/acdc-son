"""Overlay channel helpers (shared by 2D and 3D viewers)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from cellacdc.data import ImageData
from cellacdc.segmentation import experiment, io, tools


def list_sibling_channels(
    images_path: Path,
    *,
    exclude: str | None = None,
) -> list[str]:
    """Return channel names in an Images folder, optionally excluding the primary."""
    _basename, channels = experiment.discover_basename_and_channels(Path(images_path))
    if exclude is None:
        return channels
    return [name for name in channels if name != exclude]


def load_channel_image(
    images_path: Path,
    channel_name: str,
    *,
    layout: tools.StackLayout,
) -> np.ndarray:
    """Load a channel from the same Images folder and validate shape against ``layout``."""
    spec = experiment.build_load_spec(Path(images_path), channel_name)
    image = io.load_image(spec.image_path)
    primary_layout = tools.layout_from_metadata(image.shape, spec.size_t, spec.size_z)
    if (
        primary_layout.size_t != layout.size_t
        or primary_layout.size_z != layout.size_z
        or primary_layout.size_y != layout.size_y
        or primary_layout.size_x != layout.size_x
    ):
        raise ValueError(
            f'Channel "{channel_name}" shape {image.shape} does not match the primary image layout'
        )
    return image


def overlay_label(channels: Sequence[ImageData]) -> str:
    """Human-readable label for one or more overlay channels."""
    names = [
        ch.channel_name or ch.title
        for ch in channels
        if ch.channel_name or ch.title
    ]
    return " + ".join(names) if names else "overlay"


def overlay_slice_at(
    channels: Sequence[ImageData],
    layout: tools.StackLayout,
    t_index: int,
    z_index: int,
) -> np.ndarray | None:
    """Return a max-intensity composite slice for overlay channels."""
    if not channels:
        return None
    slices = [
        tools.extract_slice(ch.image, layout, t_index, z_index) for ch in channels
    ]
    return np.maximum.reduce(np.stack(slices, axis=0))


def overlay_stack_array(channels: Sequence[ImageData]) -> np.ndarray:
    """Return a max-intensity composite volume for overlay channels."""
    if not channels:
        raise ValueError("No overlay channels")
    return np.maximum.reduce(np.stack([ch.image for ch in channels], axis=0))
