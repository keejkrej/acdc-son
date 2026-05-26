"""2D segmentation viewer entry points."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar
from weakref import WeakSet

from acdc.app import exec_until_closed, get_qapp
from acdc.core.data import AcdcData, AcdcResult, coalesce_images

if TYPE_CHECKING:
    from acdc.segment.model import SegmentationModel
    from acdc.segment.presenter import SegmentationPresenter
    from acdc.segment.view import SegmentationView

_current_viewer: SegmentationViewer | None = None


class SegmentationViewer:
    """Manual segmentation viewer; owns model, view, and presenter."""

    _instances: ClassVar[WeakSet[SegmentationViewer]] = WeakSet()

    def __init__(self, *, show: bool = False) -> None:
        get_qapp()
        from acdc.segment.model import SegmentationModel
        from acdc.segment.presenter import SegmentationPresenter
        from acdc.segment.view import SegmentationView

        self._model = SegmentationModel()
        self._view = SegmentationView()
        self._presenter = SegmentationPresenter(self._model, self._view)
        self._result: AcdcResult | None = None
        self._images: tuple[AcdcData, ...] = ()
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
    def result(self) -> AcdcResult | None:
        return self._result

    @property
    def images(self) -> tuple[AcdcData, ...]:
        return self._images

    def open(
        self,
        images: Sequence[AcdcData],
        segmentation: AcdcResult,
    ) -> AcdcResult:
        """Bind ``images`` channel(s) and a live ``segmentation`` mask to this viewer."""
        image_list = coalesce_images(images)
        self._images = image_list
        self._presenter.open(list(image_list), segmentation)
        self._result = segmentation
        return segmentation

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


def run_segment(
    images: Sequence[AcdcData],
    segmentation: AcdcResult,
) -> tuple[tuple[AcdcData, ...], AcdcResult]:
    """Open the 2D segmentation editor; block until the window closes."""
    images = coalesce_images(images)
    viewer = SegmentationViewer()
    viewer.open(images, segmentation)
    viewer.show()
    exec_until_closed(viewer.view)
    return images, segmentation
