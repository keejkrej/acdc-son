"""Manual segmentation mode (MVP architecture)."""

__all__ = [
    "SegmentModel",
    "SegmentPresenter",
    "SegmentView",
    "SegmentViewer",
    "current_viewer",
]


def __getattr__(name: str):
    if name == "SegmentModel":
        from .segment_model import SegmentModel

        return SegmentModel
    if name == "SegmentPresenter":
        from .segment_presenter import SegmentPresenter

        return SegmentPresenter
    if name == "SegmentView":
        from .segment_view import SegmentView

        return SegmentView
    if name in {"SegmentViewer", "current_viewer"}:
        from . import segment_viewer as viewer_module

        return getattr(viewer_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
