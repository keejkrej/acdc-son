"""Tests for Lucide icon rendering."""

from __future__ import annotations

from acdc.ui.icons import LucideIcon, clear_icon_cache, icon_stroke_color, themed_lucide_qicon
from acdc.app import get_qapp


def test_themed_lucide_qicon_returns_non_empty_icon() -> None:
    get_qapp()
    icon = themed_lucide_qicon(LucideIcon.FOLDER_OPEN)
    assert not icon.isNull()
    pixmap = icon.pixmap(24, 24)
    assert not pixmap.isNull()
    assert pixmap.width() == 24
    assert pixmap.height() == 24


def test_icon_stroke_follows_palette() -> None:
    from qtpy.QtGui import QPalette, QColor
    from qtpy.QtWidgets import QApplication

    app = get_qapp()
    light = QPalette()
    light.setColor(QPalette.ColorRole.WindowText, QColor("#111111"))
    dark = QPalette()
    dark.setColor(QPalette.ColorRole.WindowText, QColor("#eeeeee"))

    app.setPalette(light)
    clear_icon_cache()
    assert icon_stroke_color() == "#111111"

    app.setPalette(dark)
    clear_icon_cache()
    assert icon_stroke_color() == "#eeeeee"


def test_lucide_icon_names() -> None:
    get_qapp()
    for name in (
        LucideIcon.FOLDER_OPEN,
        LucideIcon.FILE_IMAGE,
        LucideIcon.SAVE,
        LucideIcon.SAVE_AS,
        LucideIcon.UNDO,
        LucideIcon.REDO,
        LucideIcon.HAND,
        LucideIcon.MOVE,
        LucideIcon.BRUSH,
        LucideIcon.ERASER,
    ):
        icon = themed_lucide_qicon(name)
        assert not icon.isNull(), f"icon {name!r} should not be null"
