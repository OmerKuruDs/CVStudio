<div align="center">

<img src="icon.svg" alt="cvsandbox" width="140" />

# cvsandbox

**An interactive OpenCV playground. Chain image operations, tune every parameter with live preview, and export the result as ready-to-run Python.**

[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![PySide6](https://img.shields.io/badge/PySide6-6.6%2B-41CD52)](https://doc.qt.io/qtforpython/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9%2B-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-119%20passing-success)](tests/)

</div>

---

## Overview

cvsandbox is a desktop tool for **finding the right OpenCV pipeline by iterating on parameters in real time.** Load an image, stack operations (Gaussian blur → adaptive threshold → morphology → find contours, ...), tweak every parameter with auto-generated sliders, then export the entire pipeline as a self-contained Python function whose output matches the live preview byte-for-byte.

Built for engineers who currently iterate on `cv2.GaussianBlur(img, (5, 5), 0)` calls in Jupyter cells, one parameter at a time.

## Highlights

- **Live preview** with a 120 ms debounced worker thread — no UI freezes on large images.
- **24 built-in operations** across 7 categories (filtering, threshold, morphology, edge, color, geometric, analysis).
- **Auto-generated UI** — every slider, spinner, and dropdown is derived from a declarative `Parameter` spec. Adding an operation does not touch UI code.
- **Pipeline persistence** — save and load pipelines as `.cvpipe.json`.
- **Code export** — emit a stand-alone `process(img)` Python function whose output is verified equal to the live pipeline.
- **Before / after compare** — toggle a side-by-side view to see exactly what the pipeline does (`Ctrl+B`).
- **Histogram panel** — per-channel intensity overlay refreshes with every preview.
- **Zoom & pan** — cursor-anchored mouse-wheel zoom, drag-to-pan, double-click to refit.

## Quick start

```bash
git clone https://github.com/OmerKuru/cvsandbox.git
cd cvsandbox
python -m venv .venv
.venv\Scripts\activate              # Windows PowerShell
# source .venv/bin/activate         # Linux / macOS
pip install -e ".[dev]"
pytest                              # 119 tests, all passing
cvsandbox                           # launch the GUI
```

To inspect the registered operation catalog without launching the UI:

```bash
cvsandbox --list
```

## Architecture

```
src/cvsandbox/
├── core/                  # Domain primitives
│   ├── operation.py       # OperationSpec + Parameter dataclasses
│   ├── pipeline.py        # Pipeline + PipelineNode
│   ├── registry.py        # Global operation registry
│   ├── codegen.py         # Pipeline → Python source generation
│   └── serialization.py   # Load / save pipelines as JSON
├── operations/            # Built-in OpenCV operations (one module per category)
├── resources/             # Bundled assets (icon)
└── ui/                    # PySide6 widgets
    ├── app.py             # QApplication bootstrap
    ├── main_window.py     # Window assembly + menus + worker wiring
    ├── image_view.py      # Zoom / pan + before / after split
    ├── histogram_panel.py
    ├── operation_catalog.py
    ├── parameter_panel.py
    ├── parameter_widgets.py
    ├── pipeline_view.py
    ├── pipeline_worker.py # Background thread for preview
    └── code_export_dialog.py
```

Each operation is a small declarative `OperationSpec`: an id, a parameter list, a pure `(image, **params) -> image` function, and a `code_export` callable that emits matching Python. Pipelines are an ordered list of nodes; each node binds a spec to concrete parameter values. The UI introspects the parameter spec to auto-build inputs — adding a new operation does not require any UI code.

## Built-in operations

| Category    | Operations                                                              |
| ----------- | ----------------------------------------------------------------------- |
| Filtering   | Gaussian Blur, Median Blur, Bilateral Filter, NL-Means Denoise          |
| Threshold   | Binary, Otsu, Adaptive                                                  |
| Morphology  | Erode, Dilate, Open, Close                                              |
| Edge        | Canny, Sobel, Laplacian                                                 |
| Color       | To Grayscale, To HSV, Invert, Extract Channel, CLAHE, HSV In-Range Mask |
| Geometric   | Resize, Rotate, Flip                                                    |
| Analysis    | Find Contours                                                           |

## Adding a new operation

See [CONTRIBUTING.md](CONTRIBUTING.md#adding-a-new-operation) for the full recipe. Short version:

1. Add a function plus `OperationSpec` in `src/cvsandbox/operations/<category>.py`.
2. Register the module in `src/cvsandbox/operations/__init__.py`.
3. Add a test in `tests/operations/`.

The catalog, parameter panel, and code exporter all pick up the new spec automatically.

## Roadmap

**v0.1 — Core scaffold.** Registry, pipeline, serialization, CI. *Done.*

**v0.2 — Interactive MVP**
- [x] PySide6 main window with image view, pipeline list, and parameter panel
- [x] Auto-generated sliders / inputs from parameter spec
- [x] Debounced preview with a worker thread
- [x] Zoom and pan in the image view
- [x] 24 built-in operations
- [ ] Downscaling preview mode for very large images

**v0.3 — Power user**
- [x] Pipeline save / load (`.cvpipe.json`)
- [x] Code export to stand-alone Python
- [x] Histogram panel
- [x] Before / after split view
- [ ] Per-operation timing HUD
- [ ] Drag-and-drop pipeline reordering

**v1.0 and beyond.** Node-based graph UI, ROI selection, video / camera input, batch processing.

## License

Apache-2.0 — see [LICENSE](LICENSE).
