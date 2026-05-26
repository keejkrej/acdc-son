"""Per-channel weight sliders and image/segmentation crossfade."""

from __future__ import annotations

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QHBoxLayout, QLabel, QSlider, QVBoxLayout, QWidget


class BlendControlBar(QWidget):
    """Bottom-bar blend controls used by 2D and 3D viewers."""

    channel_weights_changed = Signal(list)
    image_seg_changed = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._channel_rows_layout = QVBoxLayout()
        self._channel_rows_layout.setContentsMargins(0, 0, 0, 0)
        self._channel_rows_layout.setSpacing(2)

        self._channel_rows_host = QWidget()
        self._channel_rows_host.setLayout(self._channel_rows_layout)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(2)
        layout.addWidget(self._channel_rows_host)

        self._image_seg_row = QWidget()
        image_seg_layout = QHBoxLayout(self._image_seg_row)
        image_seg_layout.setContentsMargins(0, 0, 0, 0)
        image_seg_layout.addWidget(QLabel("Images ↔ Segmentation:"))
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

        self._channel_sliders: list[QSlider] = []
        self._channel_labels: list[QLabel] = []

    def _update_image_seg_label(self, value: int) -> None:
        self._image_seg_label.setText(f"{value}%")

    def _update_channel_label(self, index: int, slider_value: int) -> None:
        self._channel_labels[index].setText(f"{slider_value / 1000:.2f}")

    def _emit_channel_weights(self) -> None:
        weights = [slider.value() / 1000.0 for slider in self._channel_sliders]
        self.channel_weights_changed.emit(weights)

    def set_channels(
        self,
        names: list[str],
        *,
        weights: list[float] | None = None,
        image_seg: int,
    ) -> None:
        while self._channel_rows_layout.count():
            item = self._channel_rows_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self._channel_sliders.clear()
        self._channel_labels.clear()

        if weights is None:
            weights = [1.0] * len(names)
        if len(weights) != len(names):
            raise ValueError("weights must match channel names")

        for index, name in enumerate(names):
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            title = name or f"Channel {index + 1}"
            row_layout.addWidget(QLabel(f"{title}:"))
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, 1000)
            slider.setValue(int(round(max(0.0, min(1.0, weights[index])) * 1000)))
            slider.valueChanged.connect(lambda _v, i=index: self._update_channel_label(i, _v))
            slider.valueChanged.connect(lambda _v: self._emit_channel_weights())
            row_layout.addWidget(slider, stretch=1)
            value_label = QLabel(f"{weights[index]:.2f}")
            value_label.setMinimumWidth(40)
            row_layout.addWidget(value_label)
            self._channel_rows_layout.addWidget(row)
            self._channel_sliders.append(slider)
            self._channel_labels.append(value_label)
            self._update_channel_label(index, slider.value())

        self._image_seg_slider.blockSignals(True)
        self._image_seg_slider.setValue(image_seg)
        self._image_seg_slider.blockSignals(False)
        self._update_image_seg_label(image_seg)

    def channel_weights(self) -> list[float]:
        return [slider.value() / 1000.0 for slider in self._channel_sliders]
