from __future__ import annotations

import numpy as np

from cvsandbox.operations.filtering import GAUSSIAN_BLUR, MEDIAN_BLUR


def test_gaussian_blur_preserves_shape_and_dtype() -> None:
    img = np.random.default_rng(0).integers(0, 255, size=(32, 32, 3), dtype=np.uint8)
    out = GAUSSIAN_BLUR.func(img, ksize=5, sigma_x=1.0)
    assert out.shape == img.shape
    assert out.dtype == img.dtype


def test_gaussian_blur_spreads_a_single_bright_pixel() -> None:
    img = np.zeros((9, 9), dtype=np.uint8)
    img[4, 4] = 255
    out = GAUSSIAN_BLUR.func(img, ksize=5, sigma_x=1.0)
    assert out[4, 4] < 255, "the bright pixel should be diffused"
    assert out[4, 3] > 0, "neighbours should pick up some intensity"


def test_gaussian_blur_snaps_even_kernel_to_odd() -> None:
    img = np.zeros((9, 9), dtype=np.uint8)
    # ksize=4 is invalid for cv2.GaussianBlur; the wrapper must coerce to 5.
    GAUSSIAN_BLUR.func(img, ksize=4, sigma_x=1.0)


def test_median_blur_removes_salt_pepper_noise() -> None:
    img = np.full((9, 9), 128, dtype=np.uint8)
    img[4, 4] = 255  # bright outlier
    img[2, 2] = 0  # dark outlier
    out = MEDIAN_BLUR.func(img, ksize=3)
    assert out[4, 4] == 128
    assert out[2, 2] == 128


def test_median_blur_snaps_even_kernel_to_odd() -> None:
    img = np.zeros((9, 9), dtype=np.uint8)
    MEDIAN_BLUR.func(img, ksize=4)
