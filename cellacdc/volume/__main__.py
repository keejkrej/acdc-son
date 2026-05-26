"""CLI entry for the 3D volume viewer."""

from __future__ import annotations

import sys

from cellacdc.viewer import run
from cellacdc.volume import VolumeViewer


def main() -> None:
    viewer = VolumeViewer(show=True)
    sys.exit(run())


if __name__ == "__main__":
    main()
