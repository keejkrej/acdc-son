"""Tests for Cell-ACDC experiment folder loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

from cellacdc.segmentation import experiment, io, tools
from cellacdc.segmentation.model import SegmentationModel
from cellacdc.segmentation.tools import layout_from_metadata


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


def test_build_load_spec_and_save_roundtrip(tmp_path: Path) -> None:
    images = _make_position(tmp_path, "Position_1", size_t=4)
    spec = experiment.build_load_spec(images, "phase")
    model = SegmentationModel()
    model.load_position(spec)
    assert model.has_data
    assert model.mask_path == images / "test_s01_segm.npz"
    sl = model.current_mask_slice()
    tools.apply_brush(sl, 8, 8, radius=2, label=1)
    model.set_mask_slice(sl)
    model.save_mask()
    loaded = io.load_mask(spec.mask_path)
    assert loaded[model.t_index, 8, 8] == 1


def test_layout_from_metadata_timelapse() -> None:
    layout = layout_from_metadata((10, 32, 32), size_t=10, size_z=1)
    assert layout.has_time and not layout.has_z
    assert layout.size_t == 10


def test_layout_from_metadata_zstack() -> None:
    layout = layout_from_metadata((8, 32, 32), size_t=1, size_z=8)
    assert layout.has_z and not layout.has_time
    assert layout.size_z == 8


def test_load_metadata_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "meta.csv"
    csv_path.write_text(
        "Description,values\nbasename,foo_\nSizeT,5\nSizeZ,1\n",
        encoding="utf-8",
    )
    meta = io.load_metadata_csv(csv_path)
    assert meta["basename"] == "foo_"
    assert meta["SizeT"] == "5"


def test_infer_image_file_context_uses_metadata(tmp_path: Path) -> None:
    images = _make_position(tmp_path, "Position_1", size_t=4)
    image_path = images / "test_s01_phase.tif"
    ctx = experiment.infer_image_file_context(image_path)
    assert ctx.size_t == 4
    assert ctx.size_z == 1
    assert ctx.basename == "test_s01_"
    assert ctx.channel_name == "phase"
    assert ctx.mask_path == images / "test_s01_segm.npz"
    assert ctx.images_path == images
    assert ctx.position_name == "Position_1"


def test_load_image_file_uses_metadata_layout(tmp_path: Path) -> None:
    images = _make_position(tmp_path, "Position_1", size_t=4, with_segm=False)
    image_path = images / "test_s01_phase.tif"
    model = SegmentationModel()
    model.load_image(image_path)
    assert model.layout is not None
    assert model.layout.has_time and not model.layout.has_z
    assert model.layout.size_t == 4
    assert model.mask_path == images / "test_s01_segm.npz"
    assert model.channel_name == "phase"


def test_load_image_file_without_metadata_uses_heuristic(tmp_path: Path) -> None:
    image_path = tmp_path / "stack.npy"
    np.save(image_path, np.zeros((8, 32, 32), dtype=np.uint16))
    model = SegmentationModel()
    model.load_image(image_path)
    assert model.layout is not None
    assert model.layout.has_z and not model.layout.has_time
    assert model.mask_path == tmp_path / "stacksegm.npz"


def test_load_metadata_voxel_sizes(tmp_path: Path) -> None:
    images = _make_position(tmp_path, "Position_1", size_z=8)
    metadata_path = images / "test_s01_metadata.csv"
    metadata_path.write_text(
        metadata_path.read_text(encoding="utf-8")
        + "PhysicalSizeZ,2.5\nPhysicalSizeY,0.5\nPhysicalSizeX,0.5\n",
        encoding="utf-8",
    )
    dz, dy, dx = experiment.load_metadata_voxel_sizes(images)
    assert dz == 2.5
    assert dy == 0.5
    assert dx == 0.5


def test_build_load_spec_includes_voxel_sizes(tmp_path: Path) -> None:
    images = _make_position(tmp_path, "Position_1")
    metadata_path = images / "test_s01_metadata.csv"
    metadata_path.write_text(
        metadata_path.read_text(encoding="utf-8") + "PhysicalSizeZ,3\nPhysicalSizeX,1\n",
        encoding="utf-8",
    )
    spec = experiment.build_load_spec(images, "phase")
    assert spec.physical_size_z == 3.0
    assert spec.physical_size_x == 1.0
