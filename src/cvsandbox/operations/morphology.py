"""Morphology operations: erode, dilate, open, close.

These accept any input shape (grayscale or color). The kernel is built from a
shape choice + odd kernel size. Iterations control how many times the op is
applied.
"""

from __future__ import annotations

import cv2
import numpy as np

from cvsandbox.core.operation import OperationSpec, Parameter

_SHAPES = {
    "Rectangle": cv2.MORPH_RECT,
    "Ellipse": cv2.MORPH_ELLIPSE,
    "Cross": cv2.MORPH_CROSS,
}


def _kernel(shape: str, ksize: int) -> np.ndarray:
    k = max(1, int(ksize) | 1)
    return cv2.getStructuringElement(_SHAPES[shape], (k, k))


def _erode(image: np.ndarray, shape: str, ksize: int, iterations: int) -> np.ndarray:
    return cv2.erode(image, _kernel(shape, ksize), iterations=int(iterations))


def _dilate(image: np.ndarray, shape: str, ksize: int, iterations: int) -> np.ndarray:
    return cv2.dilate(image, _kernel(shape, ksize), iterations=int(iterations))


def _open(image: np.ndarray, shape: str, ksize: int, iterations: int) -> np.ndarray:
    return cv2.morphologyEx(
        image, cv2.MORPH_OPEN, _kernel(shape, ksize), iterations=int(iterations)
    )


def _close(image: np.ndarray, shape: str, ksize: int, iterations: int) -> np.ndarray:
    return cv2.morphologyEx(
        image, cv2.MORPH_CLOSE, _kernel(shape, ksize), iterations=int(iterations)
    )


def _shared_params() -> tuple[Parameter, ...]:
    return (
        Parameter(
            name="shape",
            kind="choice",
            default="Rectangle",
            choices=tuple(_SHAPES.keys()),
            label="Kernel shape",
        ),
        Parameter(
            name="ksize",
            kind="kernel_size",
            default=3,
            min=1,
            max=31,
            step=2,
            label="Kernel size",
        ),
        Parameter(
            name="iterations",
            kind="int",
            default=1,
            min=1,
            max=20,
            label="Iterations",
        ),
    )


ERODE = OperationSpec(
    id="morphology.erode",
    name="Erode",
    category="Morphology",
    description="Shrinks bright regions. Good for removing small bright noise.",
    parameters=_shared_params(),
    func=_erode,
)


DILATE = OperationSpec(
    id="morphology.dilate",
    name="Dilate",
    category="Morphology",
    description="Grows bright regions. Good for joining nearby bright blobs.",
    parameters=_shared_params(),
    func=_dilate,
)


OPEN = OperationSpec(
    id="morphology.open",
    name="Open",
    category="Morphology",
    description="Erode then dilate. Removes small bright objects while preserving large shapes.",
    parameters=_shared_params(),
    func=_open,
)


CLOSE = OperationSpec(
    id="morphology.close",
    name="Close",
    category="Morphology",
    description="Dilate then erode. Closes small holes in bright regions.",
    parameters=_shared_params(),
    func=_close,
)


ALL: tuple[OperationSpec, ...] = (ERODE, DILATE, OPEN, CLOSE)
