"""Core domain types and I/O for microscopy volumes."""

from acdc.core.data import (
    AcdcData,
    AcdcResult,
    coalesce_images,
    load,
    load_segmentation,
    save_segmentation,
)
from acdc.core.stack import (
    StackShape,
    extract_slice,
    infer_shape,
    normalize_to_4d,
    shape_from_metadata,
    write_slice,
)

__all__ = [
    "AcdcData",
    "AcdcResult",
    "StackShape",
    "coalesce_images",
    "extract_slice",
    "infer_shape",
    "load",
    "load_segmentation",
    "normalize_to_4d",
    "save_segmentation",
    "shape_from_metadata",
    "write_slice",
]
