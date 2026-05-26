"""Lucide icons rendered as Qt ``QIcon`` for toolbars and menus."""

from __future__ import annotations

from functools import lru_cache

from lucide import lucide_icon
from qtpy.QtCore import QByteArray, Qt
from qtpy.QtGui import QIcon, QPainter, QPixmap
from qtpy.QtSvg import QSvgRenderer

_DEFAULT_STROKE = "#3f3f46"
_ICON_SIZES = (16, 20, 24)


class LucideIcon:
    """Lucide icon names used in the app shell."""

    FOLDER_OPEN = "folder-open"
    FILE_IMAGE = "file-image"
    SAVE = "save"
    SAVE_AS = "save-all"
    UNDO = "undo-2"
    REDO = "redo-2"
    HAND = "hand"
    MOVE = "move"
    BRUSH = "paintbrush"
    ERASER = "eraser"


def _render_svg(svg: str, size: int) -> QPixmap:
    renderer = QSvgRenderer(QByteArray(svg.encode("utf-8")))
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    renderer.render(painter)
    painter.end()
    return pixmap


@lru_cache(maxsize=128)
def lucide_qicon(name: str, stroke: str = _DEFAULT_STROKE) -> QIcon:
    """Build a multi-resolution ``QIcon`` from a Lucide icon name."""
    icon = QIcon()
    for size in _ICON_SIZES:
        svg = lucide_icon(
            name,
            width=size,
            height=size,
            stroke=stroke,
            stroke_width=2,
        )
        icon.addPixmap(_render_svg(svg, size))
    return icon
