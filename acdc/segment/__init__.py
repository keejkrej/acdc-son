"""Manual segmentation mode (MVP architecture)."""

__all__ = [
    "SegmentationModel",
    "SegmentationPresenter",
    "SegmentationView",
    "SegmentationViewer",
    "current_viewer",
]


def __getattr__(name: str):
    if name == "SegmentationModel":
        from .segment_model import SegmentationModel

        return SegmentationModel
    if name == "SegmentationPresenter":
        from .segment_presenter import SegmentationPresenter

        return SegmentationPresenter
    if name == "SegmentationView":
        from .segment_view import SegmentationView

        return SegmentationView
    if name in {"SegmentationViewer", "current_viewer"}:
        from . import segment_viewer as viewer_module

        return getattr(viewer_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
