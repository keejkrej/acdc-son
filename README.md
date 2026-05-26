# acdc-restart

Minimal **manual segmentation** GUI for microscopy, rewritten from [Cell_ACDC](https://github.com/keejkrej/Cell_ACDC) segmentation mode using **Model–View–Presenter**. No model inference, no custom Qt styling.

## Features

- Open Cell-ACDC experiment folders (`Position_n/Images/`) or individual image files
- Position and channel pickers for multi-position experiments
- Brush and eraser with configurable label ID and brush size
- Frame and Z-slice navigation for stacks
- Mask overlay toggle
- Undo / redo
- Save masks as `{basename}segm.npz` with key `arr_0` (`uint32`), compatible with Cell-ACDC
- 3D volume viewer (`acdc-3d`) with vispy overlay, segmentation-style shell (hand tool, labels dock, frame transport)

## Opening data

Primary workflow: **File → Open folder** and select any of:

- Experiment root (contains `Position_1/`, `Position_2/`, …)
- A single `Position_n/` folder
- A `Position_n/Images/` folder directly

Channel files are resolved as `{basename}{channel}_aligned.npz` (preferred) or `{basename}{channel}.tif`. Masks load/save as `{basename}segm.npz` in the same `Images/` folder.

Use **File → Open image file** for loose TIFF/NPY/NPZ outside the Cell-ACDC layout.

## Requirements

- Python 3.12 (pinned via `uv`)
- qtpy + PySide6 + pyqtgraph + vispy

## Install and run

```bash
uv sync
uv run acdc-seg    # 2D manual segmentation
uv run acdc-3d     # 3D volume viewer
```

## Programmatic API

Load data and open the 2D segmentation or 3D volume viewer without mandatory filesystem I/O:

```python
import cellacdc

image = cellacdc.ImageData.from_path("/path/to/experiment", channel="phase")
segmentation = cellacdc.SegmentationResult.empty_like(image)

# 2D manual segmentation
viewer, segmentation = cellacdc.imshow(image, segmentation)
cellacdc.run()

# 3D volume overlay (read-only)
viewer3d, segmentation = cellacdc.imshow3d(image, segmentation)
cellacdc.run()
```

- **`ImageData`** (`Experiment` / `ExperimentData` aliases) — read-only image volume + layout metadata
- **`SegmentationResult`** — label mask (`uint32`); edited in 2D, overlaid in 3D
- **`SegmentationViewer`** / **`VolumeViewer`** — core viewer objects
- **`imshow(image, segmentation, ...)` / `imshow3d(image, segmentation, ...)`** — convenience wrappers returning `(viewer, segmentation)`
- **`run()`** — Qt event loop (`uv run acdc-seg` or `uv run acdc-3d`)

3D viewer: dual LUT bars (image grey, labels viridis) and an **Image ↔ Segmentation** blend slider; vispy default volume rendering only (no exposed render controls).

## Layout

```
cellacdc/
  __init__.py              # Public API
  app.py                   # get_qapp, run
  data.py                  # ImageData + SegmentationResult
  viewer.py                # Back-compat re-exports for 2D viewer
  __main__.py              # acdc-seg CLI
  segmentation/
    viewer.py              # SegmentationViewer, imshow
    model.py               # Editing state (binds to SegmentationResult)
    view.py                # Qt / pyqtgraph UI
    presenter.py           # MVP wiring
    experiment.py          # Cell-ACDC folder discovery
    io.py                  # Cell-ACDC mask format
    tools.py               # Brush math and stack helpers
  volume/
    viewer.py              # VolumeViewer, imshow (vispy)
    __main__.py            # acdc-3d CLI
```

## License

BSD-3-Clause (see upstream Cell-ACDC).
