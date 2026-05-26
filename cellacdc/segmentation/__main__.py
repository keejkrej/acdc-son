"""CLI entry for the 2D segmentation viewer."""

from __future__ import annotations

import sys

from cellacdc.app import run
from cellacdc.segmentation.viewer import SegmentationViewer


def main() -> None:
    viewer = SegmentationViewer()
    viewer.show()
    sys.exit(run())


if __name__ == "__main__":
    main()
