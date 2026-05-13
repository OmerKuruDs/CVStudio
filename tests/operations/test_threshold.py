from __future__ import annotations

import numpy as np

from cvsandbox.operations.threshold import (
    ADAPTIVE_THRESHOLD,
    BINARY_THRESHOLD,
    OTSU_THRESHOLD,
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
