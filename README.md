# acdc-restart

Minimal **manual segmentation** GUI for microscopy, rewritten from [Cell_ACDC](https://github.com/keejkrej/Cell_ACDC) segmentation mode using **Model–View–Presenter**. No model inference, no custom Qt styling.

## Features

- Open Cell-ACDC experiment folders (`Position_n/Images/`) or individual image files
- Position and channel pickers for multi-position experiments; multi-select channel picker at load
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

Script pipelines use **`acdc.middleware`** — a Gin-style context passed through `(ctx, next)` middleware:

```python
from acdc.middleware import AcdcContext, load, segment, use, volume

ctx = load("/path/to/experiment", channels=["phase", "gfp"])

pipeline = use(segment, volume)
ctx = pipeline(ctx)

ctx.segmentation.save()  # or downstream analysis on ctx.segmentation.mask
```

Add your own steps:

```python
def normalize(ctx: AcdcContext, next_) -> None:
    # mutate ctx.images or ctx.segmentation in place
    next_()

ctx = use(normalize, segment, volume)(ctx)
```

Run a single viewer step without composing:

```python
from acdc.middleware import from_arrays, run_segment

ctx = from_arrays(images, segmentation)
ctx = run_segment(ctx)
```

Build data yourself when you need full control:

```python
from acdc.middleware import from_arrays, run_segment

images = acdc.ImageData.from_path_channels("/path/to/experiment", ["phase", "gfp"])
segmentation = acdc.SegmentationResult.empty_like(images[0])
ctx = from_arrays(images, segmentation)
ctx = run_segment(ctx)
```

- **`AcdcContext`** — mutable pipeline state: `images`, `segmentation`, optional `t_index`, `meta`
- **`load(...)`** — returns `AcdcContext`; loads mask from disk when present, otherwise new empty mask
- **`use(m1, m2, ...)`** — compose middleware; returns `ctx -> ctx`
- **`segment` / `volume`** — built-in viewer middleware (block until window closes)
- **`run_segment` / `run_volume`** — run one viewer step on a context
- **`ImageData` / `SegmentationResult`** — data types (`acdc.data`)
- **`run()`** — only for CLI apps (`uv run acdc-seg` or `uv run acdc-3d`)

3D viewer: dual LUT bars (image grey, labels viridis) and an **Image ↔ Segmentation** blend slider; vispy default volume rendering only (no exposed render controls).

## Layout

```
acdc/
  __init__.py              # Types + viewer classes
  middleware/              # Script pipeline API (AcdcContext, use, load, segment, volume)
  app.py                   # get_qapp, run
  data.py                  # ImageData + SegmentationResult + load_data
  segment/
    __main__.py            # acdc-seg CLI
    viewer.py              # SegmentationViewer, segment
    model.py               # Editing state (binds to SegmentationResult)
    view.py                # Qt / pyqtgraph UI
    presenter.py           # MVP wiring
    experiment.py          # Cell-ACDC folder discovery
    io.py                  # Cell-ACDC mask format
    tools.py               # Brush math and stack helpers
  volume/
    viewer.py              # VolumeViewer, volume (vispy)
    __main__.py            # acdc-3d CLI
```

## License

BSD-3-Clause (see upstream Cell-ACDC).
