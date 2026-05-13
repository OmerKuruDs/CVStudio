"""Filtering operations: smoothing, denoising."""

from __future__ import annotations

import cv2
import numpy as np

from cvsandbox.core.operation import OperationSpec, Parameter


def _gaussian_blur(image: np.ndarray, ksize: int, sigma_x: float) -> np.ndarray:
    k = int(ksize) | 1  # cv2 requires odd kernel sizes
    return cv2.GaussianBlur(image, (k, k), float(sigma_x))


def _median_blur(image: np.ndarray, ksize: int) -> np.ndarray:
    return cv2.medianBlur(image, int(ksize) | 1)


GAUSSIAN_BLUR = OperationSpec(
    id="filtering.gaussian_blur",
    name="Gaussian Blur",
    category="Filtering",
    description="Smooths the image with a Gaussian kernel.",
    parameters=(
        Parameter(
            name="ksize",
            kind="kernel_size",
            default=3,
            min=1,
            max=99,
            step=2,
            label="Kernel size",
            description="Odd integer; larger = blurrier.",
        ),
        Parameter(
            name="sigma_x",
            kind="float",
            default=0.0,
            min=0.0,
            max=20.0,
            step=0.1,
            label="Sigma X",
            description="Gaussian standard deviation. 0 = derive from ksize.",
        ),
    ),
    func=_gaussian_blur,
)


MEDIAN_BLUR = OperationSpec(
    id="filtering.median_blur",
    name="Median Blur",
    category="Filtering",
    description="Replaces each pixel with the median of its neighborhood. Strong against salt-and-pepper noise.",
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


ALL: tuple[OperationSpec, ...] = (GAUSSIAN_BLUR, MEDIAN_BLUR)
