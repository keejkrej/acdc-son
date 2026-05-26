"""Shared pyqtgraph LUT bar widgets."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg

LABEL_ALPHA = int(round(0.45 * 255))


def lut_with_hidden_labels(lut: np.ndarray, hidden_ids: set[int]) -> np.ndarray:
    """Return a copy of ``lut`` with ``hidden_ids`` entries fully transparent."""
    if not hidden_ids:
        return lut
    out = np.array(lut, copy=True)
    for label_id in hidden_ids:
        if 0 <= label_id < len(out):
            out[label_id] = (0, 0, 0, 0)
    return out


class BaseLutBar(pg.HistogramLUTItem):
    """Vertical LUT bar with histogram hidden (gradient only)."""

    _LABEL_STYLE = {"color": "#ffffff", "font-size": "11px"}

    def __init__(
        self,
        *,
        axis_label: str = "",
        gradient_position: str = "right",
    ) -> None:
        super().__init__(
            fillHistogram=False,
            gradientPosition=gradient_position,
            orientation="vertical",
        )
        self.vb.hide()
        self.axis.unlinkFromView()
        self.setMinimumWidth(95)
        self.setMaximumWidth(115)
        if axis_label:
            self.set_axis_label(axis_label)

    def set_axis_label(self, axis_label: str) -> None:
        self.axis.setLabel(axis_label, **self._LABEL_STYLE)
