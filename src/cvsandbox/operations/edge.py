"""Edge detection operations: Canny, Sobel, Laplacian.

Canny and Laplacian want single-channel input; Sobel works per-channel but we
also coerce to gray for a consistent, edge-map-shaped output. Sobel and
Laplacian produce signed/float arrays that we collapse back to uint8 via
`convertScaleAbs` so the pipeline keeps a stable dtype.
"""

from __future__ import annotations

import cv2
import numpy as np

from cvsandbox.core.operation import OperationSpec, Parameter


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def _canny(image: np.ndarray, threshold1: int, threshold2: int, aperture_size: int) -> np.ndarray:
    gray = _to_gray(image)
    aperture = max(3, min(7, int(aperture_size) | 1))
    return cv2.Canny(gray, float(threshold1), float(threshold2), apertureSize=aperture)


def _sobel(image: np.ndarray, dx: int, dy: int, ksize: int) -> np.ndarray:
    gray = _to_gray(image)
    # dx and dy can't both be zero — fall back to a no-op grayscale view.
    if int(dx) == 0 and int(dy) == 0:
        return gray
    k = max(1, min(7, int(ksize) | 1))
    raw = cv2.Sobel(gray, cv2.CV_64F, int(dx), int(dy), ksize=k)
    return cv2.convertScaleAbs(raw)


def _laplacian(image: np.ndarray, ksize: int) -> np.ndarray:
    gray = _to_gray(image)
    k = max(1, min(31, int(ksize) | 1))
    raw = cv2.Laplacian(gray, cv2.CV_64F, ksize=k)
    return cv2.convertScaleAbs(raw)


CANNY = OperationSpec(
    id="edge.canny",
    name="Canny",
    category="Edge",
    description="Canny edge detector. Outputs a binary edge map.",
    parameters=(
        Parameter(name="threshold1", kind="int", default=100, min=0, max=500, label="Threshold 1"),
        Parameter(name="threshold2", kind="int", default=200, min=0, max=500, label="Threshold 2"),
        Parameter(
            name="aperture_size",
            kind="kernel_size",
            default=3,
            min=3,
            max=7,
            step=2,
            label="Aperture size",
            description="Sobel kernel size used internally. Odd, 3-7.",
        ),
    ),
    func=_canny,
)


SOBEL = OperationSpec(
    id="edge.sobel",
    name="Sobel",
    category="Edge",
    description="First-order derivative. `dx`/`dy` choose the gradient direction.",
    parameters=(
        Parameter(name="dx", kind="int", default=1, min=0, max=2, label="dx"),
        Parameter(name="dy", kind="int", default=0, min=0, max=2, label="dy"),
        Parameter(
            name="ksize",
            kind="kernel_size",
            default=3,
            min=1,
            max=7,
            step=2,
            label="Kernel size",
        ),
    ),
    func=_sobel,
)


LAPLACIAN = OperationSpec(
    id="edge.laplacian",
    name="Laplacian",
    category="Edge",
    description="Second-order derivative. Highlights regions of rapid intensity change.",
    parameters=(
        Parameter(
            name="ksize",
            kind="kernel_size",
            default=3,
            min=1,
            max=31,
            step=2,
            label="Kernel size",
        ),
    ),
    func=_laplacian,
)


ALL: tuple[OperationSpec, ...] = (CANNY, SOBEL, LAPLACIAN)
