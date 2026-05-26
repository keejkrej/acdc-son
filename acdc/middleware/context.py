"""Pipeline context passed through middleware."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from acdc.core.data import AcdcData, AcdcResult


@dataclass
class AcdcContext:
    """Mutable workflow state for script pipelines."""

    images: tuple[AcdcData, ...]
    segmentation: AcdcResult
    t_index: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def primary(self) -> AcdcData:
        return self.images[0]
