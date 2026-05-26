"""3D volume viewer entry points (parallel to ``cellacdc.viewer``)."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar
from weakref import WeakSet

from cellacdc.data import ImagedData, SegmentationResult
from cellacdc.viewer import get_qapp

if TYPE_CHECKING:
    from cellacdc.volume.model import VolumeModel
    from cellacdc.volume.presenter import VolumePresenter
    from cellacdc.volume.view import VolumeView

_current_volume_viewer: VolumeViewer | None = None


class VolumeViewer:
    """Read-only 3D viewer for ``ImagedData`` with ``SegmentationResult`` overlay."""

    _instances: ClassVar[WeakSet[VolumeViewer]] = WeakSet()

    def __init__(self, *, show: bool = False) -> None:
        get_qapp()
        from cellacdc.volume.model import VolumeModel
        from cellacdc.volume.presenter import VolumePresenter
        from cellacdc.volume.view import VolumeView

        self._model = VolumeModel()
        self._view = VolumeView()
        self._presenter = VolumePresenter(self._model, self._view)
        self._instances.add(self)
        if show:
            self.show()

    @property
    def model(self) -> VolumeModel:
        return self._model

    @property
    def window(self) -> VolumeView:
        return self._view

    @property
    def presenter(self) -> VolumePresenter:
        return self._presenter

    @property
    def result(self) -> SegmentationResult | None:
        return self._model.result

    @property
    def imaged(self) -> ImagedData | None:
        return self._model.imaged

    @property
    def canvas(self):
        return self._view.canvas

    def open(
        self,
        imaged: ImagedData,
        *,
        result: SegmentationResult | None = None,
        t_index: int = 0,
    ) -> SegmentationResult:
        return self._presenter.open(imaged, result, t_index=t_index)

    def show(self) -> None:
        global _current_volume_viewer
        self._presenter.run()
        _current_volume_viewer = self

    def close(self) -> None:
        self._view.close()


def current_volume_viewer() -> VolumeViewer | None:
    """Return the most recently shown 3D volume viewer, if any."""
    return _current_volume_viewer


def imshow(
    data: ImagedData,
    *,
    result: SegmentationResult | None = None,
    viewer: VolumeViewer | None = None,
    show: bool = True,
    t_index: int = 0,
) -> tuple[VolumeViewer, SegmentationResult]:
    """Open ``data`` in the 3D volume viewer and return ``(viewer, result)``."""
    target = viewer if viewer is not None else VolumeViewer()
    mask_result = target.open(data, result=result, t_index=t_index)
    if show:
        target.show()
    return target, mask_result
