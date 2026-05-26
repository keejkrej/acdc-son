"""Backward-compatible re-exports; prefer ``cellacdc.segmentation.viewer``."""

from cellacdc.app import get_qapp, run
from cellacdc.segmentation.viewer import (
    SegmentationViewer,
    current_viewer,
    imshow,
)

__all__ = [
    "SegmentationViewer",
    "current_viewer",
    "get_qapp",
    "imshow",
    "run",
]
