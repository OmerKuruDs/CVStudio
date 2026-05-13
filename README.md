# cvsandbox

> Interactive OpenCV playground — chain image processing operations with live preview.

**Status:** Pre-alpha. Core scaffold only — UI and most operations are not implemented yet. See [Roadmap](#roadmap).

## What it is

A desktop tool for **finding the right OpenCV pipeline interactively**. Load an image, stack operations (blur → threshold → morphology → …), tweak parameters with live preview sliders, then export the pipeline as ready-to-paste Python code.

Built for engineers who currently iterate on `cv2.GaussianBlur(img, (5, 5), 0)` calls in Jupyter cells, one parameter at a time.

## Quick start (development)

```bash
git clone https://github.com/OmerKuru/cvsandbox.git
cd cvsandbox
python -m venv .venv
.venv\Scripts\activate              # Windows PowerShell
# source .venv/bin/activate         # Linux / macOS
pip install -e ".[dev]"
pytest
```

Run the (currently stub) entry point:

```bash
cvsandbox
```

## Architecture

```
src/cvsandbox/
├── core/                # Domain primitives
│   ├── operation.py     # OperationSpec + Parameter dataclasses
│   ├── pipeline.py      # Pipeline + PipelineNode
│   └── registry.py      # Global operation registry
├── operations/          # Built-in OpenCV operations
│   └── filtering.py     # GaussianBlur, MedianBlur, ...
└── ui/                  # PySide6 widgets (TODO)
```

Each operation is a small declarative `OperationSpec` (id, parameters, function). Pipelines are an ordered list of nodes; each node = spec + parameter values. The UI auto-generates sliders/inputs from the parameter spec — adding an operation does not require touching UI code.

## Adding a new operation

See [CONTRIBUTING.md](CONTRIBUTING.md#adding-a-new-operation) for the full recipe. Short version:

1. Add a function + `OperationSpec` in `src/cvsandbox/operations/<category>.py`
2. Register it in `src/cvsandbox/operations/__init__.py`
3. Add a test in `tests/operations/`

## Roadmap

**v0.1 (current):** Core scaffold, registry, pipeline, one example operation, CI.

**v0.2 — MVP:**
- PySide6 main window with image view + pipeline list + parameter panel
- Auto-generated sliders/inputs from parameter spec
- Debounced (~120 ms) preview with worker thread
- Downscaling preview mode
- 25+ built-in operations (filtering, threshold, edge, morphology, color, geometric)

**v0.3 — Power user:**
- Pipeline save/load (`.cvpipe.json`)
- Code export (Python; C++ later)
- Histogram panel + operation timing HUD
- Before/After split view

**v1.0+:** Node-based graph UI, ROI selection, video/camera input, batch processing.

## License

Apache-2.0 — see [LICENSE](LICENSE).
