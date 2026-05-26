"""Cell-ACDC experiment folder discovery and load spec building."""

from __future__ import annotations

import difflib
import re
from pathlib import Path

from acdc.core import io, metadata

POSITION_PATTERN = re.compile(r"^Position_(\d+)$", re.IGNORECASE)
SKIP_FILENAMES = metadata.SKIP_FILENAMES


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


def _is_channel_file(filename: str) -> bool:
    lower = filename.lower()
    return filename.endswith("_aligned.npz") or lower.endswith((".tif", ".tiff"))


def _basename_from_files(filenames: list[str]) -> str:
    channel_files = [name for name in filenames if _is_channel_file(name)]
    if not channel_files:
        raise ValueError(f"No channel files found in {filenames}")
    basename = channel_files[0]
    for name in channel_files:
        matcher = difflib.SequenceMatcher(None, name, basename)
        i, _j, k = matcher.find_longest_match(0, len(name), 0, len(basename))
        basename = name[i : i + k]
    return basename


def channel_name_from_file(filename: str, basename: str) -> str | None:
    if filename.endswith("_aligned.npz"):
        suffix = "_aligned.npz"
    elif filename.lower().endswith(".tif") or filename.lower().endswith(".tiff"):
        suffix = Path(filename).suffix
    else:
        return None
    if not filename.startswith(basename) or not filename.endswith(suffix):
        return None
    return filename[len(basename) : -len(suffix)]


def _discover_basename_and_channels(
    images_path: Path,
    meta: metadata.ImagesMetadata,
) -> tuple[str, list[str]]:
    filenames = _listdir_names(images_path)
    if meta.basename and meta.channels:
        existing = [
            ch
            for ch in meta.channels
            if channel_file_path(images_path, meta.basename, ch) is not None
        ]
        if existing:
            return meta.basename, sorted(set(existing))

    basename = meta.basename or _basename_from_files(filenames)
    found: set[str] = set()
    for name in filenames:
        ch = channel_name_from_file(name, basename)
        if ch:
            found.add(ch)
    if not found:
        raise ValueError(f"No channels found in {images_path}")
    return basename, sorted(found)


def discover_basename_and_channels(images_path: Path) -> tuple[str, list[str]]:
    """Discover shared basename and channel names in an Images folder."""
    images_path = Path(images_path)
    meta = metadata.read_images_metadata(images_path)
    return _discover_basename_and_channels(images_path, meta)


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


def mask_path(images_path: Path) -> Path:
    """Default segm.npz path for an Images folder."""
    basename, _channels = discover_basename_and_channels(images_path)
    return segm_file_path(images_path, basename)


def mask_path_for_image(image_path: Path) -> Path:
    """Default segm.npz path for an image file."""
    image_path = Path(image_path)
    folder = image_path.parent
    meta = metadata.read_images_metadata(folder)
    if meta.basename:
        return segm_file_path(folder, meta.basename)
    try:
        basename, _channels = discover_basename_and_channels(folder)
        return segm_file_path(folder, basename)
    except ValueError:
        return io.segm_path_for_image(image_path)


def resolve_channel(
    images_path: Path,
    channel_name: str,
) -> tuple[Path, metadata.ImagesMetadata, str | None]:
    """Return ``(image_path, metadata, position_name)`` for one channel."""
    images_path = Path(images_path)
    meta = metadata.read_images_metadata(images_path)
    basename, channels = _discover_basename_and_channels(images_path, meta)
    if channel_name not in channels:
        raise ValueError(f'Channel "{channel_name}" not found in {images_path}')
    image_path = channel_file_path(images_path, basename, channel_name)
    if image_path is None:
        raise FileNotFoundError(
            f'No image file for channel "{channel_name}" in {images_path}'
        )
    return image_path, meta, position_name_from_images_path(images_path)


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
