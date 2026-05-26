"""Read-only state for the 3D volume viewer."""

from __future__ import annotations

import numpy as np

from cellacdc.data import ImagedData, SegmentationResult, default_segmentation


class VolumeModel:
    """Holds loaded image/mask volumes for 3D display."""

    def __init__(self) -> None:
        self.imaged: ImagedData | None = None
        self.result: SegmentationResult | None = None
        self.t_index = 0
        self.label_id = 1

    @property
    def has_data(self) -> bool:
        return self.imaged is not None and self.result is not None

    def bind(self, imaged: ImagedData, result: SegmentationResult | None = None) -> SegmentationResult:
        mask = result if result is not None else default_segmentation(imaged)
        self.imaged = imaged
        self.result = mask
        self.t_index = 0
        return mask

    def all_label_ids(self) -> list[int]:
        if not self.has_data or self.result is None:
            return []
        ids = np.unique(self.result.mask)
        return sorted(int(label) for label in ids if label > 0)

    def status_label(self) -> str:
        if self.imaged is not None and self.imaged.title:
            return self.imaged.title
        if self.imaged is not None and self.imaged.image_path is not None:
            return self.imaged.image_path.name
        return ""
