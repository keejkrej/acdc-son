"""CLI entry for the 3D volume viewer."""

from __future__ import annotations

import sys

from acdc.app import run
from acdc.volume.volume_viewer import VolumeViewer


def main() -> None:
    viewer = VolumeViewer()
    viewer.show()
    sys.exit(run())


if __name__ == "__main__":
    main()
