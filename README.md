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

## Layout

```
cellacdc/
  __main__.py              # CLI entry
  segmentation/
    model.py               # State and I/O orchestration
    view.py                # Qt / pyqtgraph UI
    presenter.py           # MVP wiring
    experiment.py           # Cell-ACDC folder discovery
    io.py                  # Cell-ACDC mask format
    tools.py               # Brush math and stack helpers
```

## License

BSD-3-Clause (see upstream Cell-ACDC).
