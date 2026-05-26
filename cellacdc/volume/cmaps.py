"""Convert pyqtgraph / label LUT tables to vispy colormaps."""

from __future__ import annotations

import numpy as np


def pg_colormap_to_vispy(pg_cmap, n: int = 256):
    from vispy.color import Colormap as VisPyColormap

    colors = pg_cmap.getLookupTable(0.0, 1.0, n)
    rgba = np.asarray(colors, dtype=np.float32) / 255.0
    return VisPyColormap(rgba)


def label_lut_to_vispy(lut: np.ndarray):
    from vispy.color import Colormap as VisPyColormap

    table = np.asarray(lut, dtype=np.float32)
    if table.ndim != 2 or table.shape[1] < 3:
        raise ValueError(f"Expected labels LUT with shape (N, 4); got {table.shape}")
    if table.shape[1] == 3:
        alpha = np.ones((len(table), 1), dtype=np.float32)
        table = np.concatenate([table, alpha], axis=1)
    rgba = table[:, :4] / 255.0
    transparent = (0.0, 0.0, 0.0, 0.0)
    return VisPyColormap(
        rgba,
        interpolation="zero",
        bad_color=transparent,
        low_color=transparent,
        high_color=transparent,
    )
