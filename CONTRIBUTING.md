# Contributing to cvsandbox

Thanks for your interest. The project's main extension point is **adding new OpenCV operations** — that's the easiest way to contribute and where help is most needed.

## Development setup

```bash
python -m venv .venv
.venv\Scripts\activate              # Windows
# source .venv/bin/activate         # Linux / macOS
pip install -e ".[dev]"
pre-commit install
pytest
```

## Adding a new operation

Each operation lives in `src/cvsandbox/operations/<category>.py` and consists of two things: a pure function `(image, **params) -> image`, and an `OperationSpec` that describes its parameters.

**Example — adding `cv2.medianBlur`:**

```python
# src/cvsandbox/operations/filtering.py
import cv2
import numpy as np
from cvsandbox.core.operation import OperationSpec, Parameter


def _median_blur(image: np.ndarray, ksize: int) -> np.ndarray:
    return cv2.medianBlur(image, ksize | 1)  # ksize must be odd


MEDIAN_BLUR = OperationSpec(
    id="filtering.median_blur",
    name="Median Blur",
    category="Filtering",
    description="Replaces each pixel with the median of its neighborhood. Good for salt-and-pepper noise.",
    parameters=(
        Parameter(
            name="ksize",
            kind="kernel_size",
            default=3,
            min=1,
            max=99,
            step=2,
            label="Kernel size",
            description="Odd integer; larger = stronger denoising.",
        ),
    ),
    func=_median_blur,
)

ALL = (GAUSSIAN_BLUR, MEDIAN_BLUR)  # append your new spec here
```

Registration happens in `src/cvsandbox/operations/__init__.py` via `load_builtin_operations()`, which iterates each module's `ALL` tuple. Modules must not register at import time — keep the module side-effect free and just append to `ALL`.

**Tests:** add a corresponding `tests/operations/test_filtering.py` entry that runs the operation on a deterministic synthetic image and asserts shape + a basic property (e.g. "output mean is close to input mean", "edges are present after Canny").

## Parameter kinds

| `kind`        | UI widget         | Notes                          |
| ------------- | ----------------- | ------------------------------ |
| `int`         | Integer slider    | `min`, `max`, `step` required  |
| `float`       | Float slider      | `min`, `max`, `step` required  |
| `bool`        | Checkbox          |                                |
| `choice`      | Dropdown          | `choices` tuple required       |
| `kernel_size` | Odd-int slider    | UI snaps to odd numbers        |

## Code style

- Ruff for lint + format, mypy in strict mode. Both run on `pre-commit` and in CI.
- Public functions and `OperationSpec.description` should be in English.
- Operation functions must be **pure**: same input → same output, no global state, no I/O.

## Pull requests

- One operation (or one logical change) per PR.
- Include a test.
- Describe what OpenCV function is being wrapped and any non-obvious parameter choices.

## License

By contributing, you agree that your contributions will be licensed under the project's Apache-2.0 license.
