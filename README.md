# acdc-son

Minimal **manual segmentation** GUI for microscopy, rewritten from [Cell_ACDC](https://github.com/keejkrej/Cell_ACDC) segmentation mode using **Model–View–Presenter**. No model inference, no custom Qt styling.

## Install and run

```bash
uv sync
uv run acdc-seg    # 2D manual segmentation
uv run acdc-3d     # 3D volume viewer
```

## Programmatic API

Script pipelines use **`acdc.middleware`** — a Gin-style context passed through `(ctx, next)` middleware:

```python
from acdc.middleware import AcdcContext, load, run_segment, run_volume, use

ctx = load("/path/to/experiment", channels=["phase", "gfp"])

pipeline = use(run_segment, run_volume)
ctx = pipeline(ctx)

ctx.segmentation.save()  # or downstream analysis on ctx.segmentation.mask
```

Add your own steps:

```python
def normalize(ctx: AcdcContext, next_) -> None:
    # mutate ctx.images or ctx.segmentation in place
    next_()

ctx = use(normalize, run_segment, run_volume)(ctx)
```

Run a single viewer step without composing:

```python
from acdc.middleware import from_arrays, run_segment

ctx = from_arrays(images, segmentation)
ctx = use(run_segment)(ctx)
```

Build data yourself when you need full control:

```python
import acdc
from acdc.middleware import from_arrays, run_segment

images = acdc.AcdcData.from_experiment("/path/to/experiment", channels=["phase", "gfp"])
segmentation = acdc.AcdcResult.empty_like(images[0])
ctx = from_arrays(images, segmentation)
ctx = use(run_segment)(ctx)
```

- **`AcdcContext`** — mutable pipeline state: `images`, `segmentation`, optional `t_index`, `meta`
- **`load(...)`** — returns `AcdcContext`; loads mask from disk when present, otherwise new empty mask
- **`use(m1, m2, ...)`** — compose middleware; returns `ctx -> ctx`
- **`run_segment` / `run_volume`** — built-in viewer middleware (block until window closes)
- **`use(run_segment)(ctx)`** — run a single viewer step without composing a pipeline
- **`AcdcData` / `AcdcResult`** — data types (`acdc.core.data`)
- **`run()`** — only for CLI apps (`uv run acdc-seg` or `uv run acdc-3d`)
