"""2D segmentation viewer entry points."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar
from weakref import WeakSet

from cellacdc.app import get_qapp
from cellacdc.data import ImageData, SegmentationResult, default_segmentation

if TYPE_CHECKING:
    from cellacdc.segmentation.model import SegmentationModel
    from cellacdc.segmentation.presenter import SegmentationPresenter
    from cellacdc.segmentation.view import SegmentationView

_current_viewer: SegmentationViewer | None = None


class SegmentationViewer:
    """Manual segmentation viewer; owns model, view, and presenter."""

    _instances: ClassVar[WeakSet[SegmentationViewer]] = WeakSet()

    def __init__(self, *, show: bool = False) -> None:
        get_qapp()
        from cellacdc.segmentation.model import SegmentationModel
        from cellacdc.segmentation.presenter import SegmentationPresenter
        from cellacdc.segmentation.view import SegmentationView

        self._model = SegmentationModel()
        self._view = SegmentationView()
        self._presenter = SegmentationPresenter(self._model, self._view)
        self._result: SegmentationResult | None = None
        self._instances.add(self)
        if show:
            self.show()

    @property
    def model(self) -> SegmentationModel:
        return self._model

    @property
    def view(self) -> SegmentationView:
        return self._view

    @property
    def presenter(self) -> SegmentationPresenter:
        return self._presenter

    @property
    def result(self) -> SegmentationResult | None:
        return self._result

    def open(
        self,
        image: ImageData,
        segmentation: SegmentationResult | None = None,
    ) -> SegmentationResult:
        """Bind ``image`` and a live ``segmentation`` mask to this viewer."""
        mask_result = (
            segmentation if segmentation is not None else default_segmentation(image)
        )
        self._presenter.open(image, mask_result)
        self._result = mask_result
        return mask_result

    def show(self) -> None:
        """Show the viewer window."""
        global _current_viewer
        self._presenter.run()
        _current_viewer = self

    def close(self) -> None:
        """Close the viewer window."""
        self._view.close()


def current_viewer() -> SegmentationViewer | None:
    """Return the most recently shown segmentation viewer, if any."""
    return _current_viewer


def imshow(
    image: ImageData,
    segmentation: SegmentationResult | None = None,
    *,
    viewer: SegmentationViewer | None = None,
    show: bool = True,
) -> tuple[SegmentationViewer, SegmentationResult]:
    """Open ``image`` in the 2D segmentation viewer and return ``(viewer, segmentation)``."""
    target = viewer if viewer is not None else SegmentationViewer()
    mask_result = target.open(image, segmentation)
    if show:
        target.show()
    return target, mask_result
