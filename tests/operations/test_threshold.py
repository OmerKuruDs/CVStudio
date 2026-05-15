from __future__ import annotations

import numpy as np

from cvstudio.operations.threshold import (
    ADAPTIVE_THRESHOLD,
    BINARY_THRESHOLD,
    IN_RANGE_BGR,
    OTSU_THRESHOLD,
    TRIANGLE_THRESHOLD,
)


def _gradient_gray() -> np.ndarray:
    return np.tile(np.arange(256, dtype=np.uint8), (32, 1))  # 32x256 horizontal ramp


def _gradient_bgr() -> np.ndarray:
    gray = _gradient_gray()
    return np.stack([gray, gray, gray], axis=-1)


def test_binary_threshold_splits_at_value() -> None:
    out = BINARY_THRESHOLD.func(_gradient_gray(), thresh=127, maxval=255, inverse=False)
    assert out[0, 100] == 0  # below threshold
    assert out[0, 200] == 255  # above threshold
    assert out.ndim == 2


def test_binary_threshold_inverse_flips_output() -> None:
    out = BINARY_THRESHOLD.func(_gradient_gray(), thresh=127, maxval=255, inverse=True)
    assert out[0, 100] == 255
    assert out[0, 200] == 0


def test_binary_threshold_accepts_bgr() -> None:
    out = BINARY_THRESHOLD.func(_gradient_bgr(), thresh=127, maxval=255, inverse=False)
    assert out.ndim == 2


def test_otsu_threshold_returns_binary_mask() -> None:
    out = OTSU_THRESHOLD.func(_gradient_gray(), maxval=255, inverse=False)
    unique = set(np.unique(out).tolist())
    assert unique <= {0, 255}


def test_adaptive_threshold_handles_uneven_lighting() -> None:
    out = ADAPTIVE_THRESHOLD.func(
        _gradient_gray(),
        maxval=255,
        method="Gaussian",
        block_size=11,
        c=2,
        inverse=False,
    )
    assert out.shape == _gradient_gray().shape
    assert out.dtype == np.uint8


def test_triangle_threshold_returns_binary_mask() -> None:
    out = TRIANGLE_THRESHOLD.func(_gradient_gray(), maxval=255, inverse=False)
    assert set(np.unique(out).tolist()) <= {0, 255}
    assert out.shape == _gradient_gray().shape


def test_triangle_threshold_inverse_swaps_polarity() -> None:
    plain = TRIANGLE_THRESHOLD.func(_gradient_gray(), maxval=255, inverse=False)
    flipped = TRIANGLE_THRESHOLD.func(_gradient_gray(), maxval=255, inverse=True)
    # Inverse should be the bitwise complement.
    assert np.array_equal(flipped, 255 - plain)


def test_in_range_bgr_isolates_target_color() -> None:
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img[2:8, 2:8] = (50, 100, 200)  # rectangle of one BGR value
    out = IN_RANGE_BGR.func(
        img,
        b_low=40, b_high=60,
        g_low=90, g_high=110,
        r_low=190, r_high=210,
    )
    assert out.ndim == 2
    assert int(out[5, 5]) == 255  # inside the rect
    assert int(out[0, 0]) == 0  # outside (pure black background)


def test_in_range_bgr_swaps_low_and_high_safely() -> None:
    img = np.full((5, 5, 3), 128, dtype=np.uint8)
    # User puts low > high — op should normalise rather than crash.
    out = IN_RANGE_BGR.func(
        img,
        b_low=200, b_high=50,
        g_low=200, g_high=50,
        r_low=200, r_high=50,
    )
    assert int(out[2, 2]) == 255


def test_in_range_bgr_accepts_grayscale_input() -> None:
    img = np.full((4, 4), 128, dtype=np.uint8)
    out = IN_RANGE_BGR.func(
        img,
        b_low=100, b_high=150,
        g_low=100, g_high=150,
        r_low=100, r_high=150,
    )
    assert out.shape == (4, 4)
    assert int(out[0, 0]) == 255
