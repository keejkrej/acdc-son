"""CLI entry for the 2D segmentation viewer."""

from __future__ import annotations

import sys

from acdc.app import run
from acdc.segment.segment_viewer import SegmentViewer


def main() -> None:
    viewer = SegmentViewer()
    viewer.show()
    sys.exit(run())


if __name__ == "__main__":
    main()
