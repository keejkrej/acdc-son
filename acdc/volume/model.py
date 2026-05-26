"""Read-only state for the 3D volume viewer."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from acdc.channels import (
    default_channel_weights,
    refresh_channel_display_levels,
    resize_channel_weights,
)
from acdc.data import ImageData, SegmentationResult, coalesce_images, default_segmentation


class VolumeModel:
    """Holds loaded image/mask volumes for 3D display."""

    def __init__(self) -> None:
        self.channels: list[ImageData] = []
        self.channel_weights: list[float] = []
        self.result: SegmentationResult | None = None
        self.t_index = 0
        self.label_id = 1
        self.image_seg_blend = 50.0
        self.channel_stack_levels: list[tuple[float, float] | None] = []
        self.channel_display_clim: list[tuple[float, float] | None] = []

    @property
    def primary(self) -> ImageData | None:
        """First channel (layout reference for mask and navigation)."""
        return self.channels[0] if self.channels else None

    @property
    def has_data(self) -> bool:
        return bool(self.channels) and self.result is not None

    def bind(
        self,
        images: Sequence[ImageData],
        result: SegmentationResult | None = None,
    ) -> SegmentationResult:
        image_list = list(coalesce_images(images))
        reference = image_list[0]
        mask = result if result is not None else default_segmentation(reference)
        self.channels = image_list
        self.channel_weights = resize_channel_weights(self.channel_weights, len(image_list))
        if not self.channel_weights:
            self.channel_weights = default_channel_weights(len(image_list))
        self.result = mask
        self.t_index = 0
        self._refresh_channel_display_levels()
        return mask

    def _refresh_channel_display_levels(self) -> None:
        if not self.channels:
            self.channel_stack_levels = []
            self.channel_display_clim = []
            return
        self.channel_stack_levels, self.channel_display_clim = refresh_channel_display_levels(
            self.channels
        )

    def set_channel_weights(self, weights: Sequence[float]) -> None:
        self.channel_weights = resize_channel_weights(weights, len(self.channels))

    def set_image_seg_blend(self, value_0_to_100: float) -> None:
        self.image_seg_blend = max(0.0, min(100.0, float(value_0_to_100)))

    def all_label_ids(self) -> list[int]:
        if not self.has_data or self.result is None:
            return []
        ids = np.unique(self.result.mask)
        return sorted(int(label) for label in ids if label > 0)

    def status_label(self) -> str:
        reference = self.primary
        if reference is not None and reference.title:
            return reference.title
        if reference is not None and reference.image_path is not None:
            return reference.image_path.name
        return ""
