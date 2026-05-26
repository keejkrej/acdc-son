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

## Opening data

Primary workflow: **File → Open folder** and select any of:

- Experiment root (contains `Position_1/`, `Position_2/`, …)
- A single `Position_n/` folder
- A `Position_n/Images/` folder directly

Channel files are resolved as `{basename}{channel}_aligned.npz` (preferred) or `{basename}{channel}.tif`. Masks load/save as `{basename}segm.npz` in the same `Images/` folder.

Use **File → Open image file** for loose TIFF/NPY/NPZ outside the Cell-ACDC layout.

## Requirements

- Python 3.12 (pinned via `uv`)
- qtpy + PySide6 + pyqtgraph

## Install and run

```bash
uv sync
uv run acdc-seg
```

## Programmatic API

Load data and open the viewer without tying edits to filesystem I/O:

```python
import cellacdc

data = cellacdc.ExperimentData.from_path("/path/to/experiment", channel="phase")
result = cellacdc.SegmentationResult.empty_like(data)

viewer, result = cellacdc.imshow(data, result=result)
cellacdc.run()

# Edits are written directly into result.mask
assert result.dirty
result.save("/optional/path/segm.npz")
```

Explicit viewer handle (same code path as ``imshow``):

```python
viewer = cellacdc.SegmentationViewer()
result = viewer.open(data, result=result)
viewer.show()
cellacdc.run()
```

- **`Experiment`** (`ExperimentData` alias) — read-only image volume + layout metadata
- **`SegmentationResult`** — mutable `uint32` mask the GUI edits in place
- **`SegmentationViewer`** — core viewer object (model + view + presenter)
- **`imshow(data, result=..., viewer=..., show=True)`** — convenience wrapper; returns `(viewer, result)`
- **`run()`** — runs the Qt event loop (also used by `uv run acdc-seg`)

Fully in-memory workflow:

```python
import numpy as np
import cellacdc

image = np.zeros((128, 128), dtype=np.uint16)
data = cellacdc.Experiment.from_arrays(image, title="demo")
result = cellacdc.SegmentationResult.empty_like(data)
cellacdc.imshow(data, result=result)
cellacdc.run()
```

## Layout

```
cellacdc/
  __init__.py              # Public API
  data.py                  # Experiment + SegmentationResult types
  viewer.py                # SegmentationViewer, imshow, run
  __main__.py              # CLI entry
  segmentation/
    model.py               # Editing state (binds to SegmentationResult)
    view.py                # Qt / pyqtgraph UI
    presenter.py           # MVP wiring
    experiment.py          # Cell-ACDC folder discovery
    io.py                  # Cell-ACDC mask format
    tools.py               # Brush math and stack helpers
```

## License

BSD-3-Clause (see upstream Cell-ACDC).
