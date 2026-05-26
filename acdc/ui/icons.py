"""Lucide icons rendered as Qt ``QIcon`` for toolbars and menus."""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

from lucide import lucide_icon
from qtpy.QtCore import QByteArray, QEvent, QObject, Qt
from qtpy.QtGui import QIcon, QPainter, QPalette, QPixmap
from qtpy.QtSvg import QSvgRenderer
from qtpy.QtWidgets import QApplication, QWidget

_FALLBACK_STROKE = "#3f3f46"
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


def icon_stroke_color() -> str:
    """Return the foreground stroke color for icons from the active palette."""
    app = QApplication.instance()
    if app is None:
        return _FALLBACK_STROKE
    color = app.palette().color(QPalette.ColorRole.WindowText)
    if not color.isValid():
        return _FALLBACK_STROKE
    return color.name()


def _render_svg(svg: str, size: int) -> QPixmap:
    renderer = QSvgRenderer(QByteArray(svg.encode("utf-8")))
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    renderer.render(painter)
    painter.end()
    return pixmap


def clear_icon_cache() -> None:
    lucide_qicon.cache_clear()


@lru_cache(maxsize=128)
def lucide_qicon(name: str, stroke: str) -> QIcon:
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


def themed_lucide_qicon(name: str) -> QIcon:
    """Return a Lucide icon using the current application palette."""
    return lucide_qicon(name, icon_stroke_color())


class _IconThemeWatcher(QObject):
    def __init__(self, refresh: Callable[[], None], parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._refresh = refresh

    def _on_theme_changed(self, *_args: object) -> None:
        clear_icon_cache()
        self._refresh()

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        if event.type() in (QEvent.Type.PaletteChange, QEvent.Type.ApplicationPaletteChange):
            self._on_theme_changed()
        return super().eventFilter(watched, event)


def install_icon_theme_watcher(window: QWidget, refresh: Callable[[], None]) -> None:
    """Refresh toolbar/menu icons when the palette or color scheme changes."""
    watcher = _IconThemeWatcher(refresh, parent=window)
    window.installEventFilter(watcher)
    setattr(window, "_icon_theme_watcher", watcher)

    app = QApplication.instance()
    if app is None:
        return

    if hasattr(app, "paletteChanged"):
        app.paletteChanged.connect(watcher._on_theme_changed)

    style_hints = app.styleHints()
    if hasattr(style_hints, "colorSchemeChanged"):
        style_hints.colorSchemeChanged.connect(watcher._on_theme_changed)
