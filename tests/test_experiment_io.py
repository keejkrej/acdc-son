"""Tests for Cell-ACDC experiment folder loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

from acdc.core.data import AcdcData, AcdcResult
from acdc.core import experiment, io, metadata, stack
from acdc.core.stack import shape_from_metadata
from acdc.segment import editing
from acdc.segment.segment_model import SegmentationModel


def _write_metadata(
    images: Path,
    basename: str,
    *,
    size_t: int = 1,
    size_z: int = 1,
    channels: tuple[str, ...] = ("phase", "gfp"),
) -> None:
    lines = [
        "Description,values",
        f"basename,{basename}",
        f"SizeT,{size_t}",
        f"SizeZ,{size_z}",
    ]
    for i, ch in enumerate(channels):
        lines.append(f"channel_{i}_name,{ch}")
    (images / f"{basename}metadata.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_position(
    root: Path,
    position: str,
    basename: str = "test_s01_",
    *,
    size_t: int = 1,
    size_z: int = 1,
    with_segm: bool = True,
    aligned: bool = False,
) -> Path:
    images = root / position / "Images"
    images.mkdir(parents=True)
    data = np.zeros((4, 16, 16), dtype=np.uint16) if size_t > 1 else np.zeros((16, 16), dtype=np.uint16)
    if aligned:
        np.savez_compressed(images / f"{basename}phase_aligned.npz", arr_0=data)
        tifffile.imwrite(images / f"{basename}phase.tif", data)
    else:
        tifffile.imwrite(images / f"{basename}phase.tif", data)
        tifffile.imwrite(images / f"{basename}gfp.tif", data)
    _write_metadata(images, basename, size_t=size_t, size_z=size_z)
    if with_segm:
        mask = np.zeros_like(data, dtype=np.uint32)
        io.save_mask(images / f"{basename}segm.npz", mask)
    return images


def test_resolve_images_paths_from_experiment(tmp_path: Path) -> None:
    _make_position(tmp_path, "Position_1")
    _make_position(tmp_path, "Position_2")
    paths = experiment.resolve_images_paths(tmp_path)
    assert len(paths) == 2
    assert paths[0].name == "Images"


def test_resolve_images_paths_from_position(tmp_path: Path) -> None:
    images = _make_position(tmp_path, "Position_1")
    paths = experiment.resolve_images_paths(tmp_path / "Position_1")
    assert paths == [images]


def test_resolve_images_paths_from_images(tmp_path: Path) -> None:
    images = _make_position(tmp_path, "Position_1")
    paths = experiment.resolve_images_paths(images)
    assert paths == [images]


def test_discover_basename_and_channels_from_metadata(tmp_path: Path) -> None:
    images = _make_position(tmp_path, "Position_1")
    basename, channels = experiment.discover_basename_and_channels(images)
    assert basename == "test_s01_"
    assert channels == ["gfp", "phase"]


def test_channel_file_prefers_aligned_npz(tmp_path: Path) -> None:
    images = _make_position(tmp_path, "Position_1", aligned=True)
    path = experiment.channel_file_path(images, "test_s01_", "phase")
    assert path is not None
    assert path.name.endswith("_aligned.npz")


def test_experiment_load_and_save_roundtrip(tmp_path: Path) -> None:
    images = _make_position(tmp_path, "Position_1", size_t=4)
    imaged = AcdcData.from_experiment(images, channel="phase")[0]
    mask_path = experiment.mask_path(images)
    from acdc.core.data import AcdcResult, load_segmentation

    model = SegmentationModel()
    result = load_segmentation(mask_path, like=imaged)
    model.open([imaged], result, mask_path=mask_path)
    assert model.has_data
    assert model.mask_path == images / "test_s01_segm.npz"
    sl = model.current_mask_slice()
    editing.apply_brush(sl, 8, 8, radius=2, label=1)
    model.set_mask_slice(sl)
    model.save_mask()
    loaded = io.load_mask(mask_path)
    assert loaded[model.t_index, 8, 8] == 1


def test_shape_from_metadata_timelapse() -> None:
    stack_shape = shape_from_metadata((10, 32, 32), size_t=10, size_z=1)
    assert stack_shape.has_time and not stack_shape.has_z
    assert stack_shape.size_t == 10


def test_shape_from_metadata_zstack() -> None:
    stack_shape = shape_from_metadata((8, 32, 32), size_t=1, size_z=8)
    assert stack_shape.has_z and not stack_shape.has_time
    assert stack_shape.size_z == 8


def test_load_metadata_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "meta.csv"
    csv_path.write_text(
        "Description,values\nbasename,foo_\nSizeT,5\nSizeZ,1\n",
        encoding="utf-8",
    )
    meta = io.load_metadata_csv(csv_path)
    assert meta["basename"] == "foo_"
    assert meta["SizeT"] == "5"


def test_mask_path_for_image_uses_metadata(tmp_path: Path) -> None:
    images = _make_position(tmp_path, "Position_1", size_t=4)
    image_path = images / "test_s01_phase.tif"
    meta = metadata.read_images_metadata(images)
    assert meta.size_t == 4
    assert meta.basename == "test_s01_"
    assert experiment.channel_name_from_file(image_path.name, meta.basename) == "phase"
    assert experiment.mask_path_for_image(image_path) == images / "test_s01_segm.npz"
    assert experiment.mask_path(images) == images / "test_s01_segm.npz"
    imaged = AcdcData.from_path(image_path)
    assert imaged.name == "Position_1 / phase"


def test_open_image_file_uses_metadata_layout(tmp_path: Path) -> None:
    images = _make_position(tmp_path, "Position_1", size_t=4, with_segm=False)
    image_path = images / "test_s01_phase.tif"
    imaged = AcdcData.from_path(image_path)
    result = AcdcResult.empty_like(imaged)
    model = SegmentationModel()
    mask_path = experiment.mask_path_for_image(image_path)
    model.open([imaged], result, mask_path=mask_path)
    assert model.stack_shape is not None
    assert model.stack_shape.has_time and not model.stack_shape.has_z
    assert model.stack_shape.size_t == 4
    assert model.mask_path == images / "test_s01_segm.npz"
    assert imaged.name == "Position_1 / phase"


def test_open_image_file_without_metadata_uses_heuristic(tmp_path: Path) -> None:
    image_path = tmp_path / "stack.npy"
    np.save(image_path, np.zeros((8, 32, 32), dtype=np.uint16))
    imaged = AcdcData.from_path(image_path)
    result = AcdcResult.empty_like(imaged)
    model = SegmentationModel()
    model.open([imaged], result, mask_path=experiment.mask_path_for_image(image_path))
    assert model.stack_shape is not None
    assert model.stack_shape.has_z and not model.stack_shape.has_time
    assert model.mask_path == tmp_path / "stacksegm.npz"


def test_read_images_metadata_voxel_sizes(tmp_path: Path) -> None:
    images = _make_position(tmp_path, "Position_1", size_z=8)
    metadata_path = images / "test_s01_metadata.csv"
    metadata_path.write_text(
        metadata_path.read_text(encoding="utf-8")
        + "PhysicalSizeZ,2.5\nPhysicalSizeY,0.5\nPhysicalSizeX,0.5\n",
        encoding="utf-8",
    )
    meta = metadata.read_images_metadata(images)
    assert meta.physical_size_z == 2.5
    assert meta.physical_size_y == 0.5
    assert meta.physical_size_x == 0.5


def test_resolve_channel_includes_voxel_sizes(tmp_path: Path) -> None:
    images = _make_position(tmp_path, "Position_1")
    metadata_path = images / "test_s01_metadata.csv"
    metadata_path.write_text(
        metadata_path.read_text(encoding="utf-8") + "PhysicalSizeZ,3\nPhysicalSizeX,1\n",
        encoding="utf-8",
    )
    image_path, meta, position_name = experiment.resolve_channel(images, "phase")
    assert image_path.name == "test_s01_phase.tif"
    assert position_name == "Position_1"
    assert meta.physical_size_z == 3.0
    assert meta.physical_size_x == 1.0
    imaged = AcdcData.from_experiment(images, channel="phase")[0]
    assert imaged.physical_size_z == 3.0
    assert imaged.physical_size_x == 1.0
