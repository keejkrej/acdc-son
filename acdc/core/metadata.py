"""Cell-ACDC metadata.csv discovery and parsing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from acdc.core import io

SKIP_FILENAMES = frozenset({"desktop.ini"})


@dataclass(frozen=True)
class ImagesMetadata:
    """Parsed layout and channel metadata for an Images folder."""

    basename: str | None = None
    channels: tuple[str, ...] = ()
    size_t: int | None = None
    size_z: int | None = None
    physical_size_z: float = 1.0
    physical_size_y: float = 1.0
    physical_size_x: float = 1.0


def _basename_from_metadata_csv(path: Path) -> str | None:
    """Infer experiment basename from ``{basename}metadata.csv`` filename."""
    name = path.name
    suffix = "metadata.csv"
    if name.endswith(suffix):
        prefix = name[: -len(suffix)]
        if prefix:
            return prefix
    return None


def find_metadata_csv(folder: Path) -> Path | None:
    """Return ``*metadata.csv`` in ``folder``, if present."""
    folder = Path(folder)
    if not folder.is_dir():
        return None
    for entry in sorted(folder.iterdir()):
        if (
            entry.is_file()
            and not entry.name.startswith(".")
            and entry.name not in SKIP_FILENAMES
            and entry.name.endswith("metadata.csv")
        ):
            return entry
    return None


def read_images_metadata(folder: Path) -> ImagesMetadata:
    """Read Cell-ACDC metadata once for an Images folder (or image parent folder)."""
    metadata_path = find_metadata_csv(folder)
    if metadata_path is None:
        return ImagesMetadata()

    raw = io.load_metadata_csv(metadata_path)
    channels = tuple(
        value
        for key, value in raw.items()
        if key.startswith("channel_") and key.endswith("_name") and value
    )
    basename = raw.get("basename") or _basename_from_metadata_csv(metadata_path) or None
    return ImagesMetadata(
        basename=basename,
        channels=channels,
        size_t=_parse_int(raw.get("SizeT")),
        size_z=_parse_int(raw.get("SizeZ")),
        physical_size_z=_parse_float(raw.get("PhysicalSizeZ")) or 1.0,
        physical_size_y=_parse_float(raw.get("PhysicalSizeY")) or 1.0,
        physical_size_x=_parse_float(raw.get("PhysicalSizeX")) or 1.0,
    )


def _parse_int(value: str | None) -> int | None:
    if value is None or value.strip() == "":
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def _parse_float(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None
