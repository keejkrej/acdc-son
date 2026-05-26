"""LUT bars for the 3D volume viewer (same look as 2D segmentation)."""

from __future__ import annotations

import numpy as np

from acdc.ui.lut import LABEL_ALPHA, BaseLutBar, lut_with_hidden_labels

__all__ = ["VolumeImageLutBar", "VolumeLabelLutBar", "lut_with_hidden_labels"]


class VolumeImageLutBar(BaseLutBar):
    """LUT bar driving a vispy grayscale volume colormap."""

    def __init__(self, *, axis_label: str = "Image") -> None:
        super().__init__(axis_label=axis_label, gradient_position="right")
        self.gradient.loadPreset("grey")
        self.setLevels(0, 1)


class VolumeLabelLutBar(BaseLutBar):
    """Right LUT bar driving the vispy label overlay colormap."""

    def __init__(self) -> None:
        super().__init__(axis_label="Labels", gradient_position="left")
        self._lut_size = 2
        self.gradient.loadPreset("viridis")
        self.setLevels(0, self._lut_size - 1)

    @property
    def lut_size(self) -> int:
        return self._lut_size

    def set_lut_size(self, size: int) -> None:
        size = max(int(size), 2)
        if size == self._lut_size:
            return
        self._lut_size = size
        self.setLevels(0, self._lut_size - 1)

    def rgba_lut(self, hidden_ids: set[int] | None = None) -> np.ndarray:
        lut = np.array(
            self.gradient.getLookupTable(self._lut_size, alpha=LABEL_ALPHA),
            copy=True,
        )
        lut[0] = (0, 0, 0, 0)
        if hidden_ids:
            lut = lut_with_hidden_labels(lut, hidden_ids)
        return lut
