"""Cell-ACDC experiment folder discovery and load spec building."""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from pathlib import Path

from . import io

POSITION_PATTERN = re.compile(r"^Position_(\d+)$", re.IGNORECASE)
SKIP_FILENAMES = frozenset({"desktop.ini"})


@dataclass(frozen=True)
class PositionLoadSpec:
    """Resolved paths and metadata for one position/channel load."""

    images_path: Path
    position_name: str | None
    basename: str
    channel_name: str
    image_path: Path
    mask_path: Path
    size_t: int | None
    size_z: int | None
    physical_size_z: float = 1.0
    physical_size_y: float = 1.0
    physical_size_x: float = 1.0


@dataclass(frozen=True)
class ImageFileLoadContext:
    """Metadata and mask paths inferred from an image file's folder."""

    size_t: int | None
    size_z: int | None
    mask_path: Path
    images_path: Path | None
    position_name: str | None
    basename: str | None
    channel_name: str | None
    physical_size_z: float = 1.0
    physical_size_y: float = 1.0
    physical_size_x: float = 1.0


def _listdir_names(path: Path) -> list[str]:
    if not path.is_dir():
        return []
    return sorted(
        entry.name
        for entry in path.iterdir()
        if entry.is_file() and not entry.name.startswith(".") and entry.name not in SKIP_FILENAMES
    )


def is_position_folder(path: Path) -> bool:
    if not path.is_dir() or not POSITION_PATTERN.match(path.name):
        return False
    images = path / "Images"
    return images.is_dir() and any(images.iterdir())


def list_positions(exp_path: Path) -> list[str]:
    """Return sorted ``Position_n`` folder names under an experiment root."""
    if not exp_path.is_dir():
        return []
    names = [entry.name for entry in exp_path.iterdir() if is_position_folder(entry)]
    return sorted(names, key=lambda name: int(POSITION_PATTERN.match(name).group(1)))


def resolve_images_paths(folder: Path) -> list[Path]:
    """Normalize experiment, position, or Images path to ``Images/`` folders."""
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Path does not exist: {folder}")

    if folder.name == "Images" and folder.is_dir():
        return [folder]

    if is_position_folder(folder):
        return [folder / "Images"]

    positions = list_positions(folder)
    if positions:
        return [folder / pos / "Images" for pos in positions]

    images = folder / "Images"
    if images.is_dir() and any(images.iterdir()):
        return [images]

    raise ValueError(
        f'Not a Cell-ACDC folder (expected Position_n/Images or Images/): "{folder}"'
    )


def position_name_from_images_path(images_path: Path) -> str | None:
    parent = images_path.parent
    if POSITION_PATTERN.match(parent.name):
        return parent.name
    return None


def _metadata_csv_path(images_path: Path) -> Path | None:
    for name in _listdir_names(images_path):
        if name.endswith("metadata.csv"):
            return images_path / name
    return None


def _basename_from_files(filenames: list[str]) -> str:
    channel_files = [
        name
        for name in filenames
        if name.endswith((".tif", ".tiff", "_aligned.npz"))
    ]
    if not channel_files:
        raise ValueError(f"No channel files found in {filenames}")
    basename = channel_files[0]
    for name in channel_files:
        matcher = difflib.SequenceMatcher(None, name, basename)
        i, _j, k = matcher.find_longest_match(0, len(name), 0, len(basename))
        basename = name[i : i + k]
    return basename


def _channel_from_file(filename: str, basename: str) -> str | None:
    if filename.endswith("_aligned.npz"):
        suffix = "_aligned.npz"
    elif filename.lower().endswith(".tif") or filename.lower().endswith(".tiff"):
        suffix = Path(filename).suffix
    else:
        return None
    if not filename.startswith(basename) or not filename.endswith(suffix):
        return None
    return filename[len(basename) : -len(suffix)]


def discover_basename_and_channels(images_path: Path) -> tuple[str, list[str]]:
    """Discover shared basename and channel names in an Images folder."""
    images_path = Path(images_path)
    filenames = _listdir_names(images_path)
    metadata_path = _metadata_csv_path(images_path)
    metadata = io.load_metadata_csv(metadata_path) if metadata_path else {}

    basename = metadata.get("basename")
    channels: list[str] = []
    if metadata:
        for key, value in metadata.items():
            if key.startswith("channel_") and key.endswith("_name") and value:
                channels.append(value)

    if basename and channels:
        existing = []
        for ch in channels:
            if channel_file_path(images_path, basename, ch) is not None:
                existing.append(ch)
        if existing:
            return basename, sorted(set(existing))

    basename = basename or _basename_from_files(filenames)
    found: set[str] = set()
    for name in filenames:
        ch = _channel_from_file(name, basename)
        if ch:
            found.add(ch)
    if not found:
        raise ValueError(f"No channels found in {images_path}")
    return basename, sorted(found)


def channel_file_path(images_path: Path, basename: str, channel: str) -> Path | None:
    """Return channel file path, preferring ``_aligned.npz`` over ``.tif``."""
    images_path = Path(images_path)
    aligned = images_path / f"{basename}{channel}_aligned.npz"
    if aligned.is_file():
        return aligned
    for ext in (".tif", ".tiff"):
        tif = images_path / f"{basename}{channel}{ext}"
        if tif.is_file():
            return tif
    return None


def segm_file_path(images_path: Path, basename: str) -> Path:
    return Path(images_path) / f"{basename}{io.SEGM_SUFFIX}"


def load_metadata_layout(images_path: Path) -> tuple[int | None, int | None]:
    """Read ``SizeT`` and ``SizeZ`` from metadata.csv when present."""
    metadata_path = _metadata_csv_path(Path(images_path))
    if metadata_path is None:
        return None, None
    metadata = io.load_metadata_csv(metadata_path)
    size_t = _parse_int(metadata.get("SizeT"))
    size_z = _parse_int(metadata.get("SizeZ"))
    return size_t, size_z


def load_metadata_voxel_sizes(images_path: Path) -> tuple[float, float, float]:
    """Read ``PhysicalSizeZ/Y/X`` from metadata.csv (Cell-ACDC convention)."""
    metadata_path = _metadata_csv_path(Path(images_path))
    if metadata_path is None:
        return 1.0, 1.0, 1.0
    metadata = io.load_metadata_csv(metadata_path)
    dz = _parse_float(metadata.get("PhysicalSizeZ")) or 1.0
    dy = _parse_float(metadata.get("PhysicalSizeY")) or 1.0
    dx = _parse_float(metadata.get("PhysicalSizeX")) or 1.0
    return dz, dy, dx


def infer_image_file_context(image_path: Path) -> ImageFileLoadContext:
    """Infer layout metadata and mask path from ``metadata.csv`` in the same folder."""
    image_path = Path(image_path)
    folder = image_path.parent
    metadata_path = _metadata_csv_path(folder)
    size_t, size_z = load_metadata_layout(folder)

    basename: str | None = None
    channel_name: str | None = None
    mask_path = io.segm_path_for_image(image_path)
    images_path: Path | None = None
    position_name: str | None = None
    physical_size_z, physical_size_y, physical_size_x = load_metadata_voxel_sizes(folder)

    if metadata_path is not None:
        metadata = io.load_metadata_csv(metadata_path)
        basename = metadata.get("basename") or None
        if basename:
            channel_name = _channel_from_file(image_path.name, basename)
            mask_path = segm_file_path(folder, basename)
        images_path = folder
        if folder.name == "Images":
            position_name = position_name_from_images_path(folder)

    return ImageFileLoadContext(
        size_t=size_t,
        size_z=size_z,
        mask_path=mask_path,
        images_path=images_path,
        position_name=position_name,
        basename=basename,
        channel_name=channel_name,
        physical_size_z=physical_size_z,
        physical_size_y=physical_size_y,
        physical_size_x=physical_size_x,
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


def build_load_spec(images_path: Path, channel_name: str) -> PositionLoadSpec:
    """Build a load spec for one channel in an Images folder."""
    images_path = Path(images_path)
    basename, channels = discover_basename_and_channels(images_path)
    if channel_name not in channels:
        raise ValueError(f'Channel "{channel_name}" not found in {images_path}')
    image_path = channel_file_path(images_path, basename, channel_name)
    if image_path is None:
        raise FileNotFoundError(
            f'No image file for channel "{channel_name}" in {images_path}'
        )
    size_t, size_z = load_metadata_layout(images_path)
    physical_size_z, physical_size_y, physical_size_x = load_metadata_voxel_sizes(images_path)
    return PositionLoadSpec(
        images_path=images_path,
        position_name=position_name_from_images_path(images_path),
        basename=basename,
        channel_name=channel_name,
        image_path=image_path,
        mask_path=segm_file_path(images_path, basename),
        size_t=size_t,
        size_z=size_z,
        physical_size_z=physical_size_z,
        physical_size_y=physical_size_y,
        physical_size_x=physical_size_x,
    )


def images_path_for_position(
    folder: Path,
    images_paths: list[Path],
    position_name: str,
) -> Path:
    """Resolve a picked position name to its ``Images/`` folder."""
    direct = Path(folder) / position_name / "Images"
    if direct in images_paths:
        return direct
    if direct.is_dir():
        return direct
    for path in images_paths:
        if position_name_from_images_path(path) == position_name:
            return path
    raise ValueError(f'Position "{position_name}" not found under "{folder}"')


def resolve_folder_load(folder: Path, channel_name: str | None = None) -> PositionLoadSpec:
    """Resolve folder to a single ``PositionLoadSpec`` (one Images folder)."""
    images_paths = resolve_images_paths(folder)
    if len(images_paths) != 1:
        raise ValueError("Multiple positions require explicit selection")
    images_path = images_paths[0]
    _basename, channels = discover_basename_and_channels(images_path)
    chosen = channel_name or channels[0]
    return build_load_spec(images_path, chosen)
