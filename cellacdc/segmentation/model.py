"""Segmentation model: image/mask state (no Qt)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from cellacdc.overlay import SecondaryChannel

from . import experiment, io, tools

if TYPE_CHECKING:
    from cellacdc.data import ImagedData, SegmentationResult


class SegmentationModel:
    """Holds experiment data and editing state for manual segmentation."""

    def __init__(self) -> None:
        self.image: np.ndarray | None = None
        self.mask: np.ndarray | None = None
        self.layout: tools.StackLayout | None = None
        self.image_path: Path | None = None
        self.mask_path: Path | None = None
        self.images_path: Path | None = None
        self.position_name: str | None = None
        self.basename: str | None = None
        self.channel_name: str | None = None
        self.title: str = ""
        self.t_index = 0
        self.z_index = 0
        self.brush_size = 4
        self.label_id = 1
        self.tool = "hand"  # "move" | "hand" | "brush" | "eraser"
        self._result: SegmentationResult | None = None
        self._dirty = False
        self._undo: list[np.ndarray] = []
        self._redo: list[np.ndarray] = []
        self._stroke_snapshot: np.ndarray | None = None
        self._last_paint_y: int | None = None
        self._last_paint_x: int | None = None
        self.secondary: SecondaryChannel | None = None
        self.primary_secondary_blend = 50.0
        self.image_seg_blend = 50.0
        self.image_display_levels: tuple[float, float] | None = None
        self.secondary_display_levels: tuple[float, float] | None = None

    @property
    def dirty(self) -> bool:
        if self._result is not None:
            return self._result.dirty
        return self._dirty

    @dirty.setter
    def dirty(self, value: bool) -> None:
        if self._result is not None:
            self._result.dirty = value
        else:
            self._dirty = value

    @property
    def result(self) -> SegmentationResult | None:
        return self._result

    @property
    def has_data(self) -> bool:
        return self.image is not None and self.mask is not None and self.layout is not None

    def open(self, imaged: ImagedData, result: SegmentationResult) -> None:
        """Bind in-memory image input and a live segmentation result."""
        if result.mask.shape != imaged.image.shape:
            raise ValueError(
                f"Mask shape {result.mask.shape} does not match "
                f"image shape {imaged.image.shape}"
            )
        self.image = imaged.image
        self.mask = result.mask
        self.layout = imaged.layout
        self._result = result
        self.image_path = imaged.image_path
        self.mask_path = result.save_path or imaged.mask_path
        self.images_path = imaged.images_path
        self.position_name = imaged.position_name
        self.basename = imaged.basename
        self.channel_name = imaged.channel_name
        self.title = imaged.title
        self.t_index = 0
        self.z_index = 0
        self.secondary = None
        self.secondary_display_levels = None
        self.dirty = False
        self._clear_history()
        self._refresh_image_display_levels()

    def _clear_experiment_context(self) -> None:
        self.images_path = None
        self.position_name = None
        self.basename = None
        self.channel_name = None
        self.title = ""

    def _load_arrays(
        self,
        image: np.ndarray,
        layout: tools.StackLayout,
        image_path: Path,
        mask_path: Path,
        *,
        images_path: Path | None = None,
        position_name: str | None = None,
        basename: str | None = None,
        channel_name: str | None = None,
        title: str = "",
    ) -> None:
        if mask_path.is_file():
            mask = io.load_mask(mask_path)
            if mask.shape != image.shape:
                mask = io.empty_mask_like(image)
            saved_mask_path = mask_path
        else:
            mask = io.empty_mask_like(image)
            saved_mask_path = mask_path

        self._result = None
        self.image = image
        self.mask = mask
        self.layout = layout
        self.image_path = image_path
        self.mask_path = saved_mask_path
        self.images_path = images_path
        self.position_name = position_name
        self.basename = basename
        self.channel_name = channel_name
        self.title = title
        self.t_index = 0
        self.z_index = 0
        self.secondary = None
        self.secondary_display_levels = None
        self.dirty = False
        self._clear_history()
        self._refresh_image_display_levels()

    def load_image(self, path: Path) -> None:
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
        self._load_arrays(
            image,
            layout,
            path,
            ctx.mask_path,
            images_path=ctx.images_path,
            position_name=ctx.position_name,
            basename=ctx.basename,
            channel_name=ctx.channel_name,
            title=title,
        )

    def load_position(self, spec: experiment.PositionLoadSpec) -> None:
        image = io.load_image(spec.image_path)
        layout = tools.layout_from_metadata(image.shape, spec.size_t, spec.size_z)
        if spec.position_name and spec.channel_name:
            title = f"{spec.position_name} / {spec.channel_name}"
        else:
            title = spec.image_path.name
        self._load_arrays(
            image,
            layout,
            spec.image_path,
            spec.mask_path,
            images_path=spec.images_path,
            position_name=spec.position_name,
            basename=spec.basename,
            channel_name=spec.channel_name,
            title=title,
        )

    def save_mask(self, path: Path | None = None) -> Path:
        if not self.has_data or self.mask is None:
            raise RuntimeError("No mask loaded")
        if self._result is not None:
            dest = self._result.save(path)
            self.mask_path = dest
            return dest
        dest = Path(path) if path is not None else self.mask_path
        if dest is None:
            raise RuntimeError("No mask path set")
        io.save_mask(dest, self.mask)
        self.mask_path = dest
        self.dirty = False
        return dest

    def status_label(self) -> str:
        if self.title:
            return self.title
        if self.position_name and self.channel_name:
            return f"{self.position_name} / {self.channel_name}"
        if self.image_path is not None:
            return self.image_path.name
        return ""

    def current_image_slice(self) -> np.ndarray:
        assert self.image is not None and self.layout is not None
        return tools.extract_slice(self.image, self.layout, self.t_index, self.z_index)

    def current_mask_slice(self) -> np.ndarray:
        assert self.mask is not None and self.layout is not None
        return tools.extract_slice(self.mask, self.layout, self.t_index, self.z_index)

    def current_secondary_slice(self) -> np.ndarray | None:
        if self.secondary is None or self.layout is None:
            return None
        return self.secondary.slice_at(self.layout, self.t_index, self.z_index)

    def secondary_sibling_channels(self) -> list[str]:
        if self.images_path is None:
            return []
        from cellacdc.overlay import list_sibling_channels

        return list_sibling_channels(self.images_path, exclude=self.channel_name)

    def load_secondary_channel(self, channel_name: str) -> None:
        if not self.images_path or self.layout is None:
            raise ValueError("Secondary channel requires a Cell-ACDC Images folder")
        from cellacdc.overlay import load_channel_image

        image = load_channel_image(
            self.images_path,
            channel_name,
            layout=self.layout,
        )
        self.secondary = SecondaryChannel(channel_name, image)
        self._refresh_secondary_display_levels()

    def clear_secondary(self) -> None:
        self.secondary = None
        self.secondary_display_levels = None

    def _refresh_image_display_levels(self) -> None:
        if self.image is None or self.layout is None:
            self.image_display_levels = None
            return
        from cellacdc.display_levels import stack_autoscale_levels

        self.image_display_levels = stack_autoscale_levels(self.image, self.layout)

    def _refresh_secondary_display_levels(self) -> None:
        if self.secondary is None or self.layout is None:
            self.secondary_display_levels = None
            return
        from cellacdc.display_levels import stack_autoscale_levels

        self.secondary_display_levels = stack_autoscale_levels(
            self.secondary.image,
            self.layout,
        )

    def set_primary_secondary_blend(self, value_0_to_100: float) -> None:
        self.primary_secondary_blend = max(0.0, min(100.0, float(value_0_to_100)))

    def set_image_seg_blend(self, value_0_to_100: float) -> None:
        self.image_seg_blend = max(0.0, min(100.0, float(value_0_to_100)))

    def current_label_ids(self) -> list[int]:
        """Return sorted unique label IDs on the current slice (excluding background)."""
        if not self.has_data:
            return []
        ids = np.unique(self.current_mask_slice())
        return sorted(int(label) for label in ids if label > 0)

    def label_at(self, y: int, x: int) -> int:
        """Return the label ID at ``(y, x)`` on the current slice (0 if background)."""
        if not self.has_data:
            return 0
        sl = self.current_mask_slice()
        h, w = sl.shape
        if not (0 <= y < h and 0 <= x < w):
            return 0
        return int(sl[y, x])

    def labels_in_rect(self, y0: int, x0: int, y1: int, x1: int) -> list[int]:
        """Return label IDs intersecting an inclusive image rectangle."""
        if not self.has_data:
            return []
        return tools.unique_labels_in_rect(
            self.current_mask_slice(),
            y0,
            x0,
            y1,
            x1,
        )

    def all_label_ids(self) -> list[int]:
        """Return sorted unique label IDs in the full mask (excluding background)."""
        if not self.has_data or self.mask is None:
            return []
        ids = np.unique(self.mask)
        return sorted(int(label) for label in ids if label > 0)

    def max_label_id(self) -> int:
        """Return the highest label ID in the full mask volume."""
        if self.mask is None:
            return 0
        return int(self.mask.max())

    def set_mask_slice(self, slice_2d: np.ndarray) -> None:
        assert self.mask is not None and self.layout is not None
        self.mask = tools.write_slice(
            self.mask, self.layout, self.t_index, self.z_index, slice_2d
        )
        self.dirty = True

    def begin_stroke(self) -> None:
        if self.mask is not None:
            self._stroke_snapshot = self.mask.copy()
        self._last_paint_y = None
        self._last_paint_x = None

    def end_stroke(self) -> None:
        if self._stroke_snapshot is None or self.mask is None:
            self._last_paint_y = None
            self._last_paint_x = None
            return
        if self.tool == "brush" and self.layout is not None:
            sl = tools.extract_slice(self.mask, self.layout, self.t_index, self.z_index)
            if tools.fill_label_holes(sl, self.label_id):
                self.dirty = True
        if not np.array_equal(self._stroke_snapshot, self.mask):
            self._undo.append(self._stroke_snapshot)
            if len(self._undo) > 30:
                self._undo.pop(0)
            self._redo.clear()
        self._stroke_snapshot = None
        self._last_paint_y = None
        self._last_paint_x = None

    def undo(self) -> bool:
        if not self._undo or self.mask is None:
            return False
        self._redo.append(self.mask.copy())
        self.mask = self._undo.pop()
        self.dirty = True
        return True

    def redo(self) -> bool:
        if not self._redo or self.mask is None:
            return False
        self._undo.append(self.mask.copy())
        self.mask = self._redo.pop()
        self.dirty = True
        return True

    def paint(self, y: int, x: int) -> None:
        if not self.has_data or self.mask is None or self.layout is None:
            return
        sl = tools.extract_slice(self.mask, self.layout, self.t_index, self.z_index)
        tools.apply_brush_stroke(
            sl,
            y,
            x,
            self._last_paint_y,
            self._last_paint_x,
            self.brush_size,
            self.label_id,
            erase=self.tool == "eraser",
        )
        self._last_paint_y = y
        self._last_paint_x = x
        self.dirty = True

    def _clear_history(self) -> None:
        self._undo.clear()
        self._redo.clear()
        self._stroke_snapshot = None
