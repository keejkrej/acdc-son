"""Shared label list dock widget."""

from __future__ import annotations

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QAbstractItemView,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)


class LabelListPanel(QWidget):
    """Explorer-style label list with global and per-label visibility toggles."""

    label_id_changed = Signal(int)
    label_visibility_changed = Signal()
    labels_selected = Signal(list)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._updating = False
        self._hidden_labels: set[int] = set()
        self._all_label_ids: list[int] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(0)

        self._list = QListWidget()
        self._list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._list.itemChanged.connect(self._on_item_changed)
        self._list.currentItemChanged.connect(self._on_current_item_changed)
        self._list.itemSelectionChanged.connect(self._on_item_selection_changed)
        layout.addWidget(self._list)

    def reset_visibility(self) -> None:
        self._hidden_labels.clear()

    def get_hidden_label_ids(self) -> set[int]:
        return set(self._hidden_labels)

    def set_label_list(
        self,
        label_ids: list[int],
        *,
        selected_ids: list[int] | None = None,
    ) -> None:
        self._all_label_ids = list(label_ids)
        selected = set(selected_ids or [])

        self._list.blockSignals(True)
        self._updating = True
        self._list.clear()
        for label_id in label_ids:
            item = QListWidgetItem(str(label_id))
            item.setData(Qt.UserRole, label_id)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            checked = label_id not in self._hidden_labels
            item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
            self._list.addItem(item)
        self._updating = False

        self._list.clearSelection()
        for row in range(self._list.count()):
            item = self._list.item(row)
            if item is None:
                continue
            label = item.data(Qt.UserRole)
            if label is not None and int(label) in selected:
                item.setSelected(True)

        if selected:
            for row in range(self._list.count()):
                item = self._list.item(row)
                if item is not None and item.isSelected():
                    self._list.setCurrentRow(row)
                    break
        else:
            self._list.setCurrentRow(-1)

        self._list.blockSignals(False)

    def set_active_label(self, label_id: int) -> None:
        self._select_label(label_id)

    def set_selection_state(
        self,
        label_ids: list[int],
        *,
        active_id: int | None = None,
    ) -> None:
        ids = set(label_ids)
        self._list.blockSignals(True)
        self._list.clearSelection()
        for row in range(self._list.count()):
            item = self._list.item(row)
            if item is None:
                continue
            label = item.data(Qt.UserRole)
            if label is not None and int(label) in ids:
                item.setSelected(True)
        if active_id is not None:
            self._select_label(active_id)
        elif not label_ids:
            self._list.setCurrentRow(-1)
        self._list.blockSignals(False)

    def _on_item_selection_changed(self) -> None:
        if self._updating:
            return
        label_ids: list[int] = []
        for item in self._list.selectedItems():
            label_id = item.data(Qt.UserRole)
            if label_id is not None:
                label_ids.append(int(label_id))
        self.labels_selected.emit(sorted(label_ids))

    def _select_label(self, label_id: int) -> None:
        self._list.blockSignals(True)
        for row in range(self._list.count()):
            item = self._list.item(row)
            if item is not None and item.data(Qt.UserRole) == label_id:
                self._list.setCurrentRow(row)
                break
        self._list.blockSignals(False)

    def _on_item_changed(self, item: QListWidgetItem) -> None:
        if self._updating:
            return
        label_id = item.data(Qt.UserRole)
        if label_id is None:
            return
        label_id = int(label_id)
        if item.checkState() == Qt.Checked:
            self._hidden_labels.discard(label_id)
        else:
            self._hidden_labels.add(label_id)
        self.label_visibility_changed.emit()

    def _on_current_item_changed(
        self,
        current: QListWidgetItem | None,
        _previous: QListWidgetItem | None,
    ) -> None:
        if current is None:
            return
        label_id = current.data(Qt.UserRole)
        if label_id is not None:
            self.label_id_changed.emit(int(label_id))
