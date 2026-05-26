"""Segmentation viewer and programmatic entry points (napari-style)."""

from __future__ import annotations

import os
import warnings
from typing import TYPE_CHECKING, ClassVar
from weakref import WeakSet

from cellacdc.data import ImagedData, SegmentationResult, default_segmentation

if TYPE_CHECKING:
    from qtpy.QtWidgets import QApplication

    from cellacdc.segmentation.model import SegmentationModel
    from cellacdc.segmentation.presenter import SegmentationPresenter
    from cellacdc.segmentation.view import SegmentationView

_qt_configured = False
_current_viewer: SegmentationViewer | None = None


def get_qapp() -> QApplication:
    """Return the shared Qt application, creating it when needed."""
    global _qt_configured
    os.environ.setdefault("QT_API", "pyside6")
    if not _qt_configured:
        import pyqtgraph as pg

        pg.setConfigOptions(imageAxisOrder="row-major")
        _qt_configured = True
    from qtpy.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


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
        imaged: ImagedData,
        *,
        result: SegmentationResult | None = None,
    ) -> SegmentationResult:
        """Bind ``imaged`` and a live ``result`` mask to this viewer."""
        mask_result = result if result is not None else default_segmentation(imaged)
        self._presenter.open(imaged, mask_result)
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
    data: ImagedData,
    *,
    result: SegmentationResult | None = None,
    viewer: SegmentationViewer | None = None,
    show: bool = True,
) -> tuple[SegmentationViewer, SegmentationResult]:
    """Open ``data`` in a viewer and return ``(viewer, result)``."""
    target = viewer if viewer is not None else SegmentationViewer()
    mask_result = target.open(data, result=result)
    if show:
        target.show()
    return target, mask_result


def run(*, force: bool = False) -> int:
    """Start the Qt event loop."""
    app = get_qapp()
    if not force and not app.topLevelWidgets():
        warnings.warn(
            "No top-level widgets are visible; call viewer.show() first or use run(force=True).",
            RuntimeWarning,
            stacklevel=2,
        )
        return 0
    return app.exec()
