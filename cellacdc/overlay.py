"""Optional secondary channel overlay (shared by 2D and 3D viewers)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cellacdc.segmentation import experiment, io, tools


@dataclass
class SecondaryChannel:
    """Second imaging channel aligned to the primary image volume."""

    channel_name: str
    image: np.ndarray

    def slice_at(self, layout: tools.StackLayout, t_index: int, z_index: int) -> np.ndarray:
        return tools.extract_slice(self.image, layout, t_index, z_index)


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
