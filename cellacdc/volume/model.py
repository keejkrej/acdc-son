"""Read-only state for the 3D volume viewer."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from cellacdc.data import ImageData, SegmentationResult, coalesce_images, default_segmentation
from cellacdc.display_levels import stack_display_levels


class VolumeModel:
    """Holds loaded image/mask volumes for 3D display."""

    def __init__(self) -> None:
        self.primary: ImageData | None = None
        self.overlay_channels: list[ImageData] = []
        self.result: SegmentationResult | None = None
        self.t_index = 0
        self.label_id = 1
        self.primary_secondary_blend = 50.0
        self.image_seg_blend = 50.0
        self.primary_stack_levels: tuple[float, float] | None = None
        self.primary_display_clim: tuple[float, float] | None = None
        self.secondary_stack_levels: tuple[float, float] | None = None
        self.secondary_display_clim: tuple[float, float] | None = None

    @property
    def has_data(self) -> bool:
        return self.primary is not None and self.result is not None

    def bind(
        self,
        images: Sequence[ImageData],
        result: SegmentationResult | None = None,
    ) -> SegmentationResult:
        image_list = coalesce_images(images)
        primary = image_list[0]
        mask = result if result is not None else default_segmentation(primary)
        self.primary = primary
        self.overlay_channels = list(image_list[1:])
        self.result = mask
        self.t_index = 0
        self._refresh_primary_display_levels()
        self._refresh_secondary_display_levels()
        return mask

    def _refresh_primary_display_levels(self) -> None:
        if self.primary is None:
            self.primary_stack_levels = None
            self.primary_display_clim = None
            return
        (self.primary_stack_levels, self.primary_display_clim) = stack_display_levels(
            self.primary.image,
            self.primary.layout,
        )

    def _refresh_secondary_display_levels(self) -> None:
        if not self.overlay_channels or self.primary is None:
            self.secondary_stack_levels = None
            self.secondary_display_clim = None
            return
        from cellacdc.overlay import overlay_stack_array

        overlay = overlay_stack_array(self.overlay_channels)
        (self.secondary_stack_levels, self.secondary_display_clim) = stack_display_levels(
            overlay,
            self.primary.layout,
        )

    def set_primary_secondary_blend(self, value_0_to_100: float) -> None:
        self.primary_secondary_blend = max(0.0, min(100.0, float(value_0_to_100)))

    def set_image_seg_blend(self, value_0_to_100: float) -> None:
        self.image_seg_blend = max(0.0, min(100.0, float(value_0_to_100)))

    def all_label_ids(self) -> list[int]:
        if not self.has_data or self.result is None:
            return []
        ids = np.unique(self.result.mask)
        return sorted(int(label) for label in ids if label > 0)

    def status_label(self) -> str:
        if self.primary is not None and self.primary.title:
            return self.primary.title
        if self.primary is not None and self.primary.image_path is not None:
            return self.primary.image_path.name
        return ""
