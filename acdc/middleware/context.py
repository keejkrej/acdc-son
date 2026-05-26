"""Pipeline context passed through middleware."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from acdc.data import ImageData, SegmentationResult


@dataclass
class AcdcContext:
    """Mutable workflow state for script pipelines."""

    images: tuple[ImageData, ...]
    segmentation: SegmentationResult
    t_index: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def primary(self) -> ImageData:
        return self.images[0]
