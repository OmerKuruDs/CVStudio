"""Threshold operations.

All thresholds need a single-channel input; 3-channel input is converted to
grayscale at the boundary. Output is single-channel uint8 — downstream ops that
need 3 channels should add a grayscale-to-BGR conversion after.
"""

from __future__ import annotations

import cv2
import numpy as np

from cvsandbox.core.operation import OperationSpec, Parameter


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def _binary_threshold(image: np.ndarray, thresh: int, maxval: int, inverse: bool) -> np.ndarray:
    gray = _to_gray(image)
    mode = cv2.THRESH_BINARY_INV if inverse else cv2.THRESH_BINARY
    _, out = cv2.threshold(gray, float(thresh), float(maxval), mode)
    return out


def _otsu_threshold(image: np.ndarray, maxval: int, inverse: bool) -> np.ndarray:
    gray = _to_gray(image)
    mode = (cv2.THRESH_BINARY_INV if inverse else cv2.THRESH_BINARY) | cv2.THRESH_OTSU
    _, out = cv2.threshold(gray, 0, float(maxval), mode)
    return out


_ADAPTIVE_METHODS = {
    "Mean": cv2.ADAPTIVE_THRESH_MEAN_C,
    "Gaussian": cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
}


def _adaptive_threshold(
    image: np.ndarray,
    maxval: int,
    method: str,
    block_size: int,
    c: int,
    inverse: bool,
) -> np.ndarray:
    gray = _to_gray(image)
    block = max(3, int(block_size) | 1)  # must be odd, >= 3
    mode = cv2.THRESH_BINARY_INV if inverse else cv2.THRESH_BINARY
    return cv2.adaptiveThreshold(
        gray,
        float(maxval),
        _ADAPTIVE_METHODS[method],
        mode,
        block,
        float(c),
    )


BINARY_THRESHOLD = OperationSpec(
    id="threshold.binary",
    name="Binary Threshold",
    category="Threshold",
    description="Pixels above `thresh` become `maxval`, others become 0.",
    parameters=(
        Parameter(name="thresh", kind="int", default=127, min=0, max=255, label="Threshold"),
        Parameter(name="maxval", kind="int", default=255, min=0, max=255, label="Max value"),
        Parameter(name="inverse", kind="bool", default=False, label="Invert"),
    ),
    func=_binary_threshold,
)


OTSU_THRESHOLD = OperationSpec(
    id="threshold.otsu",
    name="Otsu Threshold",
    category="Threshold",
    description="Picks the threshold automatically from the image histogram (bimodal assumption).",
    parameters=(
        Parameter(name="maxval", kind="int", default=255, min=0, max=255, label="Max value"),
        Parameter(name="inverse", kind="bool", default=False, label="Invert"),
    ),
    func=_otsu_threshold,
)


ADAPTIVE_THRESHOLD = OperationSpec(
    id="threshold.adaptive",
    name="Adaptive Threshold",
    category="Threshold",
    description="Threshold computed per-region. Handles uneven illumination.",
    parameters=(
        Parameter(name="maxval", kind="int", default=255, min=0, max=255, label="Max value"),
        Parameter(
            name="method",
            kind="choice",
            default="Gaussian",
            choices=tuple(_ADAPTIVE_METHODS.keys()),
            label="Method",
        ),
        Parameter(
            name="block_size",
            kind="kernel_size",
            default=11,
            min=3,
            max=99,
            step=2,
            label="Block size",
            description="Odd, ≥3. Neighborhood used to compute the local threshold.",
        ),
        Parameter(
            name="c",
            kind="int",
            default=2,
            min=-50,
            max=50,
            label="C",
            description="Constant subtracted from the local mean.",
        ),
        Parameter(name="inverse", kind="bool", default=False, label="Invert"),
    ),
    func=_adaptive_threshold,
)


ALL: tuple[OperationSpec, ...] = (BINARY_THRESHOLD, OTSU_THRESHOLD, ADAPTIVE_THRESHOLD)
