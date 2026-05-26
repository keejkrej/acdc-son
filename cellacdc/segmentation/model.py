"""Segmentation model: image/mask state and persistence (no Qt)."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np

from . import io, tools


class SegmentationModel:
    """Holds experiment data and editing state for manual segmentation."""

    def __init__(self) -> None:
        self.image: np.ndarray | None = None
        self.mask: np.ndarray | None = None
        self.layout: tools.StackLayout | None = None
        self.image_path: Path | None = None
        self.mask_path: Path | None = None
        self.t_index = 0
        self.z_index = 0
        self.brush_size = 4
        self.label_id = 1
        self.tool = "brush"  # "brush" | "eraser"
        self.dirty = False
        self._undo: list[np.ndarray] = []
        self._redo: list[np.ndarray] = []
        self._stroke_snapshot: np.ndarray | None = None
        self._last_paint_y: int | None = None
        self._last_paint_x: int | None = None

    @property
    def has_data(self) -> bool:
        return self.image is not None and self.mask is not None and self.layout is not None

    def load_image(self, path: Path) -> None:
        path = Path(path)
        image = io.load_image(path)
        layout = tools.infer_layout(image.shape)
        mask_path = io.segm_path_for_image(path)
        if mask_path.is_file():
            mask = io.load_mask(mask_path)
            if mask.shape != image.shape:
                mask = io.empty_mask_like(image)
            saved_mask_path = mask_path
        else:
            mask = io.empty_mask_like(image)
            saved_mask_path = mask_path

        self.image = image
        self.mask = mask
        self.layout = layout
        self.image_path = path
        self.mask_path = saved_mask_path
        self.t_index = 0
        self.z_index = 0
        self.dirty = False
        self._clear_history()

    def save_mask(self, path: Path | None = None) -> Path:
        if not self.has_data or self.mask is None:
            raise RuntimeError("No mask loaded")
        dest = Path(path) if path is not None else self.mask_path
        if dest is None:
            raise RuntimeError("No mask path set")
        io.save_mask(dest, self.mask)
        self.mask_path = dest
        self.dirty = False
        return dest

    def current_image_slice(self) -> np.ndarray:
        assert self.image is not None and self.layout is not None
        return tools.extract_slice(self.image, self.layout, self.t_index, self.z_index)

    def current_mask_slice(self) -> np.ndarray:
        assert self.mask is not None and self.layout is not None
        return tools.extract_slice(self.mask, self.layout, self.t_index, self.z_index)

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
