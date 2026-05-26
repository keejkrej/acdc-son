"""Entry point for the manual segmentation GUI."""

from __future__ import annotations

import sys

from cellacdc.viewer import SegmentationViewer, run


def main() -> None:
    viewer = SegmentationViewer()
    viewer.show()
    sys.exit(run())


if __name__ == "__main__":
    main()
