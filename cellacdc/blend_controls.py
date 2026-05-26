"""Shared primary/secondary and image/segmentation crossfade sliders."""

from __future__ import annotations

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QHBoxLayout, QLabel, QSlider, QVBoxLayout, QWidget


class BlendControlBar(QWidget):
    """Bottom-bar crossfade controls used by 2D and 3D viewers."""

    primary_secondary_changed = Signal(int)
    image_seg_changed = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(2)

        self._image_seg_row = QWidget()
        image_seg_layout = QHBoxLayout(self._image_seg_row)
        image_seg_layout.setContentsMargins(0, 0, 0, 0)
        image_seg_layout.addWidget(QLabel("(Primary/Secondary) ↔ Segmentation:"))
        self._image_seg_slider = QSlider(Qt.Horizontal)
        self._image_seg_slider.setRange(0, 100)
        self._image_seg_slider.setValue(50)
        self._image_seg_slider.valueChanged.connect(self.image_seg_changed.emit)
        image_seg_layout.addWidget(self._image_seg_slider, stretch=1)
        self._image_seg_label = QLabel("50%")
        self._image_seg_label.setMinimumWidth(48)
        image_seg_layout.addWidget(self._image_seg_label)
        self._image_seg_slider.valueChanged.connect(self._update_image_seg_label)
        layout.addWidget(self._image_seg_row)

        self._primary_secondary_row = QWidget()
        primary_secondary_layout = QHBoxLayout(self._primary_secondary_row)
        primary_secondary_layout.setContentsMargins(0, 0, 0, 0)
        self._primary_secondary_title = QLabel("Primary ↔ Secondary:")
        primary_secondary_layout.addWidget(self._primary_secondary_title)
        self._primary_secondary_slider = QSlider(Qt.Horizontal)
        self._primary_secondary_slider.setRange(0, 100)
        self._primary_secondary_slider.setValue(50)
        self._primary_secondary_slider.valueChanged.connect(self.primary_secondary_changed.emit)
        primary_secondary_layout.addWidget(self._primary_secondary_slider, stretch=1)
        self._primary_secondary_label = QLabel("50%")
        self._primary_secondary_label.setMinimumWidth(48)
        primary_secondary_layout.addWidget(self._primary_secondary_label)
        self._primary_secondary_slider.valueChanged.connect(self._update_primary_secondary_label)
        layout.addWidget(self._primary_secondary_row)
        self._primary_secondary_row.setVisible(False)

    def _update_image_seg_label(self, value: int) -> None:
        self._image_seg_label.setText(f"{value}%")

    def _update_primary_secondary_label(self, value: int) -> None:
        self._primary_secondary_label.setText(f"{value}%")

    def set_values(
        self,
        *,
        primary_secondary: int,
        image_seg: int,
        show_primary_secondary: bool,
        channel_name: str = "",
    ) -> None:
        self._primary_secondary_row.setVisible(show_primary_secondary)
        if show_primary_secondary:
            title = (
                f"Primary ↔ Secondary ({channel_name}):"
                if channel_name
                else "Primary ↔ Secondary:"
            )
            self._primary_secondary_title.setText(title)
            self._primary_secondary_slider.blockSignals(True)
            self._primary_secondary_slider.setValue(primary_secondary)
            self._primary_secondary_slider.blockSignals(False)
            self._update_primary_secondary_label(primary_secondary)
        self._image_seg_slider.blockSignals(True)
        self._image_seg_slider.setValue(image_seg)
        self._image_seg_slider.blockSignals(False)
        self._update_image_seg_label(image_seg)
