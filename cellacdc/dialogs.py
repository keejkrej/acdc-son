"""Shared Qt dialogs."""

from __future__ import annotations

from qtpy.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QListWidget,
    QVBoxLayout,
    QWidget,
)


def pick_from_list(parent: QWidget, title: str, names: list[str]) -> str | None:
    """Show a list picker dialog and return the selected name."""
    if not names:
        return None
    dialog = QDialog(parent)
    dialog.setWindowTitle(title)
    layout = QVBoxLayout(dialog)
    layout.addWidget(QLabel(f"{title}:"))
    list_widget = QListWidget()
    list_widget.addItems(names)
    list_widget.setCurrentRow(0)
    list_widget.itemDoubleClicked.connect(dialog.accept)
    layout.addWidget(list_widget)
    buttons = QDialogButtonBox(
        QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
        parent=dialog,
    )
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    if dialog.exec() != QDialog.Accepted:
        return None
    selected = list_widget.currentItem()
    return selected.text() if selected is not None else None


def pick_many_from_list(
    parent: QWidget,
    title: str,
    names: list[str],
    *,
    min_selection: int = 1,
) -> list[str] | None:
    """Show a multi-select list picker and return the chosen names."""
    if not names:
        return None
    dialog = QDialog(parent)
    dialog.setWindowTitle(title)
    layout = QVBoxLayout(dialog)
    layout.addWidget(QLabel(f"{title} (select one or more):"))
    list_widget = QListWidget()
    list_widget.setSelectionMode(QListWidget.ExtendedSelection)
    list_widget.addItems(names)
    list_widget.setCurrentRow(0)
    layout.addWidget(list_widget)
    buttons = QDialogButtonBox(
        QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
        parent=dialog,
    )
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    if dialog.exec() != QDialog.Accepted:
        return None
    selected = [item.text() for item in list_widget.selectedItems()]
    if len(selected) < min_selection:
        return None
    return selected
