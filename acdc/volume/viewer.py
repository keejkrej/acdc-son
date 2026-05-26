"""3D volume viewer entry points."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar
from weakref import WeakSet

from acdc.app import exec_until_closed, get_qapp
from acdc.core.data import AcdcData, AcdcResult, coalesce_images

if TYPE_CHECKING:
    from acdc.volume.model import VolumeModel
    from acdc.volume.presenter import VolumePresenter
    from acdc.volume.view import VolumeView

_current_volume_viewer: VolumeViewer | None = None


class VolumeViewer:
    """Read-only 3D viewer for ``AcdcData`` with ``AcdcResult`` overlay."""

    _instances: ClassVar[WeakSet[VolumeViewer]] = WeakSet()

    def __init__(self, *, show: bool = False) -> None:
        get_qapp()
        from acdc.volume.model import VolumeModel
        from acdc.volume.presenter import VolumePresenter
        from acdc.volume.view import VolumeView

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
    def view(self) -> VolumeView:
        return self._view

    @property
    def presenter(self) -> VolumePresenter:
        return self._presenter

    @property
    def result(self) -> AcdcResult | None:
        return self._model.result

    @property
    def primary(self) -> AcdcData | None:
        return self._model.primary

    @property
    def images(self) -> tuple[AcdcData, ...]:
        return tuple(self._model.channels)

    @property
    def canvas(self):
        return self._view.canvas

    def open(
        self,
        images: Sequence[AcdcData],
        segmentation: AcdcResult,
        *,
        t_index: int = 0,
    ) -> AcdcResult:
        return self._presenter.open(images, segmentation, t_index=t_index)

    def show(self) -> None:
        global _current_volume_viewer
        self._presenter.run()
        _current_volume_viewer = self

    def close(self) -> None:
        self._view.close()


def current_volume_viewer() -> VolumeViewer | None:
    """Return the most recently shown 3D volume viewer, if any."""
    return _current_volume_viewer


def run_volume(
    images: Sequence[AcdcData],
    segmentation: AcdcResult,
    *,
    t_index: int = 0,
) -> tuple[tuple[AcdcData, ...], AcdcResult]:
    """Open the 3D volume viewer; block until the window closes."""
    images = coalesce_images(images)
    viewer = VolumeViewer()
    viewer.open(images, segmentation, t_index=t_index)
    viewer.show()
    exec_until_closed(viewer.view)
    return images, segmentation
