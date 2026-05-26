"""2D segmentation LUT bars linked to pyqtgraph image items."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg

from acdc.ui.lut import LABEL_ALPHA, BaseLutBar

_DEFAULT_LUT_SIZE = 2


class ImageLutBar(BaseLutBar):
    """Left LUT bar linked to a grayscale microscopy channel."""

    def __init__(self, image_item: pg.ImageItem, *, axis_label: str = "Image") -> None:
        super().__init__(axis_label=axis_label, gradient_position="right")
        self.gradient.loadPreset("grey")
        self.setImageItem(image_item)


class SegmentationLutBar(BaseLutBar):
    """Right LUT bar controlling the label overlay colormap."""

    def __init__(self, mask_item: pg.ImageItem) -> None:
        super().__init__(axis_label="Labels", gradient_position="left")
        self._mask_item = mask_item
        self._lut_size = _DEFAULT_LUT_SIZE
        self.gradient.loadPreset("viridis")
        self.gradient.sigGradientChanged.connect(self._apply_lut)
        self.setLevels(0, self._lut_size - 1)
        self._apply_lut()

    def ensure_lut_size(self, min_size: int) -> None:
        if min_size <= self._lut_size:
            return
        self._lut_size = min_size
        self.setLevels(0, self._lut_size - 1)
        self._apply_lut()

    @property
    def lut_size(self) -> int:
        return self._lut_size

    def current_lut(self) -> np.ndarray:
        lut = np.array(
            self.gradient.getLookupTable(self._lut_size, alpha=LABEL_ALPHA),
            copy=True,
        )
        lut[0] = (0, 0, 0, 0)
        return lut

    def set_label_display_max(self, max_label_id: int) -> None:
        """Size the LUT and histogram range to the highest label ID in view."""
        display_max = max(int(max_label_id), 1)
        self.ensure_lut_size(display_max + 1)
        self.setLevels(0, display_max)
        self._apply_lut()

    def _apply_lut(self) -> None:
        lut = np.array(
            self.gradient.getLookupTable(self._lut_size, alpha=LABEL_ALPHA),
            copy=True,
        )
        lut[0] = (0, 0, 0, 0)
        self._mask_item.setLookupTable(lut)
        lo, hi = self.getLevels()
        self._mask_item.setLevels([lo, hi])
