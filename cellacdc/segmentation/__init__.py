"""Manual segmentation mode (MVP architecture)."""

__all__ = [
    "SegmentationModel",
    "SegmentationPresenter",
    "SegmentationView",
    "SegmentationViewer",
    "current_viewer",
    "imshow",
]


def __getattr__(name: str):
    if name == "SegmentationModel":
        from .model import SegmentationModel

        return SegmentationModel
    if name == "SegmentationPresenter":
        from .presenter import SegmentationPresenter

        return SegmentationPresenter
    if name == "SegmentationView":
        from .view import SegmentationView

        return SegmentationView
    if name in {"SegmentationViewer", "current_viewer", "imshow"}:
        from . import viewer as viewer_module

        return getattr(viewer_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
