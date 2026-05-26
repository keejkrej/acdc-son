"""Segmentation model: image/mask state (no Qt)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from acdc.core import io, stack
from acdc.segment import editing

if TYPE_CHECKING:
    from acdc.core.data import AcdcData, AcdcResult


class SegmentationModel:
    """Holds experiment data and editing state for manual segmentation."""

    def __init__(self) -> None:
        self.image: np.ndarray | None = None
        self.mask: np.ndarray | None = None
        self.stack_shape: stack.StackShape | None = None
        self.mask_path: Path | None = None
        self.title: str = ""
        self.t_index = 0
        self.z_index = 0
        self.brush_size = 4
        self.label_id = 1
        self.tool = "hand"  # "move" | "hand" | "brush" | "eraser"
        self._result: AcdcResult | None = None
        self.saved = True
        self._undo: list[np.ndarray] = []
        self._redo: list[np.ndarray] = []
        self._stroke_snapshot: np.ndarray | None = None
        self._last_paint_y: int | None = None
        self._last_paint_x: int | None = None
        self.channels: list[AcdcData] = []
        self.channel_weights: list[float] = []
        self.image_seg_blend = 50.0
        self.channel_display_levels: list[tuple[float, float] | None] = []

    @property
    def result(self) -> AcdcResult | None:
        return self._result

    @property
    def has_data(self) -> bool:
        return self.image is not None and self.mask is not None and self.stack_shape is not None

    def open(
        self,
        images: Sequence[AcdcData],
        result: AcdcResult,
        *,
        mask_path: Path | None = None,
    ) -> None:
        """Bind in-memory image channel(s) and a live segmentation result."""
        from acdc.utils.channels import default_channel_weights, resize_channel_weights
        from acdc.core.data import coalesce_images

        image_list = list(coalesce_images(images))
        imaged = image_list[0]
        if result.mask.shape != imaged.image.shape:
            raise ValueError(
                f"Mask shape {result.mask.shape} does not match "
                f"image shape {imaged.image.shape}"
            )
        self.image = imaged.image
        self.mask = result.mask
        self.stack_shape = imaged.stack_shape
        self._result = result
        self.mask_path = mask_path
        self.title = imaged.name
        self.t_index = 0
        self.z_index = 0
        self.channels = image_list
        self.channel_weights = resize_channel_weights(self.channel_weights, len(image_list))
        if not self.channel_weights:
            self.channel_weights = default_channel_weights(len(image_list))
        self.saved = True
        self._clear_history()
        self._refresh_channel_display_levels()

    def save_mask(self, path: Path | None = None) -> Path:
        if not self.has_data or self.mask is None:
            raise RuntimeError("No mask loaded")
        dest = Path(path) if path is not None else self.mask_path
        if dest is None:
            raise RuntimeError("No mask path set")
        io.save_mask(dest, self.mask)
        self.mask_path = dest
        self.saved = True
        return dest

    def status_label(self) -> str:
        return self.title

    def current_image_slice(self) -> np.ndarray:
        assert self.image is not None and self.stack_shape is not None
        return stack.extract_slice(self.image, self.stack_shape, self.t_index, self.z_index)

    def current_mask_slice(self) -> np.ndarray:
        assert self.mask is not None and self.stack_shape is not None
        return stack.extract_slice(self.mask, self.stack_shape, self.t_index, self.z_index)

    def current_channel_slice(self, index: int) -> np.ndarray:
        assert self.stack_shape is not None
        channel = self.channels[index]
        return stack.extract_slice(channel.image, self.stack_shape, self.t_index, self.z_index)

    def current_channel_slices(self) -> list[np.ndarray]:
        return [self.current_channel_slice(i) for i in range(len(self.channels))]

    def _refresh_channel_display_levels(self) -> None:
        if not self.channels or self.stack_shape is None:
            self.channel_display_levels = []
            return
        from acdc.utils.display_levels import stack_autoscale_levels

        self.channel_display_levels = [
            stack_autoscale_levels(channel.image, self.stack_shape) for channel in self.channels
        ]

    def set_channel_weights(self, weights: Sequence[float]) -> None:
        from acdc.utils.channels import resize_channel_weights

        self.channel_weights = resize_channel_weights(weights, len(self.channels))

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
        return editing.unique_labels_in_rect(
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
        assert self.mask is not None and self.stack_shape is not None
        self.mask = stack.write_slice(
            self.mask, self.stack_shape, self.t_index, self.z_index, slice_2d
        )
        self.saved = False

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
        if self.tool == "brush" and self.stack_shape is not None:
            sl = stack.extract_slice(self.mask, self.stack_shape, self.t_index, self.z_index)
            if editing.fill_label_holes(sl, self.label_id):
                self.saved = False
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
        self.saved = False
        return True

    def redo(self) -> bool:
        if not self._redo or self.mask is None:
            return False
        self._undo.append(self.mask.copy())
        self.mask = self._redo.pop()
        self.saved = False
        return True

    def paint(self, y: int, x: int) -> None:
        if not self.has_data or self.mask is None or self.stack_shape is None:
            return
        sl = stack.extract_slice(self.mask, self.stack_shape, self.t_index, self.z_index)
        editing.apply_brush_stroke(
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
        self.saved = False

    def _clear_history(self) -> None:
        self._undo.clear()
        self._redo.clear()
        self._stroke_snapshot = None
