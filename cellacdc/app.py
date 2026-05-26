"""Shared Qt application helpers for Cell-ACDC viewers."""

from __future__ import annotations

import os
import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qtpy.QtWidgets import QApplication

_qt_configured = False


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
