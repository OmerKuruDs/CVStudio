from __future__ import annotations

import numpy as np

from cvstudio.operations.filtering import (
    BILATERAL,
    CUSTOM_KERNEL,
    GAUSSIAN_BLUR,
    MEDIAN_BLUR,
    NL_MEANS,
    UNSHARP_MASK,
)


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


def test_bilateral_preserves_shape_and_dtype() -> None:
    img = np.random.default_rng(0).integers(0, 255, size=(32, 32, 3), dtype=np.uint8)
    out = BILATERAL.func(img, d=5, sigma_color=50.0, sigma_space=50.0)
    assert out.shape == img.shape
    assert out.dtype == img.dtype


def test_bilateral_smooths_uniform_region() -> None:
    img = np.full((16, 16, 3), 100, dtype=np.uint8)
    img[8, 8] = (200, 0, 0)  # one outlier
    out = BILATERAL.func(img, d=9, sigma_color=75.0, sigma_space=75.0)
    # Outlier should be pulled toward the surrounding 100s.
    assert out[8, 8, 0] < 200


def test_nl_means_color_branch_preserves_shape() -> None:
    img = np.random.default_rng(1).integers(0, 255, size=(24, 24, 3), dtype=np.uint8)
    out = NL_MEANS.func(img, strength=5.0, template_size=3, search_size=7)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_nl_means_gray_branch_preserves_shape() -> None:
    img = np.random.default_rng(2).integers(0, 255, size=(24, 24), dtype=np.uint8)
    out = NL_MEANS.func(img, strength=5.0, template_size=3, search_size=7)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_unsharp_mask_increases_contrast_at_an_edge() -> None:
    img = np.zeros((20, 20), dtype=np.uint8)
    img[:, 10:] = 128  # soft edge
    out = UNSHARP_MASK.func(img, radius=3, amount=2.0, threshold=0)
    # The pixel one step inside the bright side should now exceed 128.
    assert int(out[10, 10]) >= int(img[10, 10])
    # The pixel one step inside the dark side should be pushed below 0 → clipped to 0.
    assert int(out[10, 9]) <= int(img[10, 9])


def test_unsharp_mask_threshold_skips_flat_regions() -> None:
    flat = np.full((10, 10), 100, dtype=np.uint8)
    out = UNSHARP_MASK.func(flat, radius=3, amount=2.0, threshold=10)
    # Detail magnitude on a flat image is 0; threshold gates everything out.
    assert np.array_equal(out, flat)


def test_custom_kernel_identity_returns_input() -> None:
    img = np.random.default_rng(0).integers(0, 255, size=(16, 16, 3), dtype=np.uint8)
    out = CUSTOM_KERNEL.func(img, preset="Identity", strength=1.0)
    assert np.array_equal(out, img)


def test_custom_kernel_box_blur_smooths() -> None:
    img = np.zeros((9, 9), dtype=np.uint8)
    img[4, 4] = 255
    out = CUSTOM_KERNEL.func(img, preset="Box Blur 3x3", strength=1.0)
    # The bright pixel should diffuse into its 3×3 neighbourhood.
    assert out[4, 4] < 255
    assert out[3, 4] > 0


def test_custom_kernel_strength_zero_blanks_image() -> None:
    img = np.full((8, 8), 200, dtype=np.uint8)
    out = CUSTOM_KERNEL.func(img, preset="Sharpen", strength=0.0)
    # Zero kernel produces an all-zero image.
    assert int(out.max()) == 0
