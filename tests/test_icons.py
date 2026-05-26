"""Tests for Lucide icon rendering."""

from __future__ import annotations

from cellacdc.icons import LucideIcon, lucide_qicon
from cellacdc.viewer import get_qapp


def test_lucide_qicon_returns_non_empty_icon() -> None:
    get_qapp()
    icon = lucide_qicon(LucideIcon.FOLDER_OPEN)
    assert not icon.isNull()
    pixmap = icon.pixmap(24, 24)
    assert not pixmap.isNull()
    assert pixmap.width() == 24
    assert pixmap.height() == 24


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
        icon = lucide_qicon(name)
        assert not icon.isNull(), f"icon {name!r} should not be null"
