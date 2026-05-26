"""Public data types for programmatic use (decoupled from the GUI)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cellacdc.segmentation import experiment, io, tools


@dataclass(frozen=True)
class ImageData:
    """Read-only microscopy image volume plus layout metadata."""

    image: np.ndarray
    layout: tools.StackLayout
    title: str = ""
    image_path: Path | None = None
    mask_path: Path | None = None
    images_path: Path | None = None
    position_name: str | None = None
    basename: str | None = None
    channel_name: str | None = None
    physical_size_z: float = 1.0
    physical_size_y: float = 1.0
    physical_size_x: float = 1.0

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        position: str | None = None,
        channel: str | None = None,
    ) -> ImageData:
        """Load from a Cell-ACDC experiment, position, Images, or image file path."""
        path = Path(path)
        if path.is_file():
            return cls.from_image_path(path)

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

        basename, channels = experiment.discover_basename_and_channels(images_path)
        chosen = channel or channels[0]
        if channel is not None and channel not in channels:
            raise ValueError(
                f'Channel "{channel}" not found; available: {", ".join(channels)}'
            )
        spec = experiment.build_load_spec(images_path, chosen)
        image = io.load_image(spec.image_path)
        layout = tools.layout_from_metadata(image.shape, spec.size_t, spec.size_z)
        if spec.position_name and spec.channel_name:
            title = f"{spec.position_name} / {spec.channel_name}"
        else:
            title = spec.image_path.name
        return cls(
            image=image,
            layout=layout,
            title=title,
            image_path=spec.image_path,
            mask_path=spec.mask_path,
            images_path=spec.images_path,
            position_name=spec.position_name,
            basename=spec.basename,
            channel_name=spec.channel_name,
            physical_size_z=spec.physical_size_z,
            physical_size_y=spec.physical_size_y,
            physical_size_x=spec.physical_size_x,
        )

    @classmethod
    def from_image_path(cls, path: str | Path) -> ImageData:
        """Load a single TIFF/NPY/NPZ file outside a full experiment tree."""
        path = Path(path)
        image = io.load_image(path)
        ctx = experiment.infer_image_file_context(path)
        layout = tools.layout_from_metadata(image.shape, ctx.size_t, ctx.size_z)
        if ctx.position_name and ctx.channel_name:
            title = f"{ctx.position_name} / {ctx.channel_name}"
        elif ctx.channel_name:
            title = f"{path.name} ({ctx.channel_name})"
        else:
            title = path.name
        return cls(
            image=image,
            layout=layout,
            title=title,
            image_path=path,
            mask_path=ctx.mask_path,
            images_path=ctx.images_path,
            position_name=ctx.position_name,
            basename=ctx.basename,
            channel_name=ctx.channel_name,
            physical_size_z=ctx.physical_size_z,
            physical_size_y=ctx.physical_size_y,
            physical_size_x=ctx.physical_size_x,
        )

    @classmethod
    def from_arrays(
        cls,
        image: np.ndarray,
        *,
        size_t: int | None = None,
        size_z: int | None = None,
        title: str = "",
    ) -> ImageData:
        """Wrap an in-memory array (no filesystem)."""
        image = np.asarray(image)
        layout = tools.layout_from_metadata(image.shape, size_t, size_z)
        return cls(image=image, layout=layout, title=title or "array")

    @classmethod
    def from_path_channels(
        cls,
        path: str | Path,
        channels: Sequence[str],
        *,
        position: str | None = None,
    ) -> list[ImageData]:
        """Load one or more channels from a Cell-ACDC path."""
        if not channels:
            raise ValueError("At least one channel is required")
        loaded = [cls.from_path(path, position=position, channel=ch) for ch in channels]
        coalesce_images(loaded)
        return loaded


def coalesce_images(images: Sequence[ImageData]) -> tuple[ImageData, ...]:
    """Validate a channel list and return it as a tuple."""
    seq = tuple(images)
    if not seq:
        raise ValueError("At least one ImageData is required")
    primary = seq[0]
    for other in seq[1:]:
        if other.image.shape != primary.image.shape:
            raise ValueError(
                f"Channel shape {other.image.shape} does not match "
                f"primary shape {primary.image.shape}"
            )
        if other.layout != primary.layout:
            raise ValueError("All channels must share the same stack layout")
    return seq


class SegmentationResult:
    """Label mask volume; editable in 2D segmentation, overlay-only in 3D."""

    def __init__(
        self,
        mask: np.ndarray,
        *,
        save_path: Path | None = None,
    ) -> None:
        self.mask = np.asarray(mask, dtype=np.uint32)
        self.save_path = Path(save_path) if save_path is not None else None
        self.dirty = False

    @classmethod
    def empty_like(cls, imaged: ImageData) -> SegmentationResult:
        return cls(
            io.empty_mask_like(imaged.image),
            save_path=imaged.mask_path,
        )

    @classmethod
    def from_path(cls, path: str | Path, *, like: ImageData) -> SegmentationResult:
        path = Path(path)
        mask = io.load_mask(path)
        if mask.shape != like.image.shape:
            raise ValueError(
                f"Mask shape {mask.shape} does not match image shape {like.image.shape}"
            )
        return cls(mask, save_path=path)

    def save(self, path: str | Path | None = None) -> Path:
        """Persist the mask to NPZ (Cell-ACDC ``arr_0`` convention)."""
        dest = Path(path) if path is not None else self.save_path
        if dest is None:
            raise RuntimeError("No save path provided")
        io.save_mask(dest, self.mask)
        self.save_path = dest
        self.dirty = False
        return dest


def default_segmentation(imaged: ImageData) -> SegmentationResult:
    """Return an on-disk mask when present, otherwise an empty mask."""
    if imaged.mask_path is not None and imaged.mask_path.is_file():
        try:
            return SegmentationResult.from_path(imaged.mask_path, like=imaged)
        except ValueError:
            pass
    return SegmentationResult.empty_like(imaged)

