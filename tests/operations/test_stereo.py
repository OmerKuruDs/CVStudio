from __future__ import annotations

import cv2
import numpy as np

from cvstudio.operations.stereo import STEREO_BM, STEREO_SGBM


def _stereo_pair(
    width: int = 240, height: int = 120, shift: int = 12
) -> tuple[np.ndarray, np.ndarray]:
    """A high-contrast left/right pair with a clear horizontal shift.

    The image has enough texture (vertical stripes + a centred rectangle)
    that BM/SGBM both succeed; right view is the left view shifted left
    by ``shift`` pixels, simulating disparity from a closer object.
    """
    left = np.zeros((height, width), dtype=np.uint8)
    # Vertical stripes — adds texture for matching.
    left[:, ::8] = 200
    # Bright rectangle in the middle.
    cv2.rectangle(left, (width // 2 - 40, height // 2 - 30),
                  (width // 2 + 40, height // 2 + 30), 255, thickness=-1)
    # Right view = left translated to the left (positive disparity).
    right = np.zeros_like(left)
    right[:, : width - shift] = left[:, shift:]
    return left, right


# ---------------------------------------------------------------- Stereo BM


def test_stereo_bm_returns_uint8_singlechannel() -> None:
    left, right = _stereo_pair()
    out = STEREO_BM.func(left, right, num_disparities=64, block_size=15)
    assert out.dtype == np.uint8
    assert out.ndim == 2
    assert out.shape == left.shape


def test_stereo_bm_rounds_num_disparities_to_multiple_of_16() -> None:
    left, right = _stereo_pair()
    # 70 is not a multiple of 16 — the op must round it (to 64) without raising.
    out = STEREO_BM.func(left, right, num_disparities=70, block_size=15)
    assert out.shape == left.shape


def test_stereo_bm_accepts_bgr_inputs() -> None:
    left, right = _stereo_pair()
    bgr_l = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
    bgr_r = cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)
    out = STEREO_BM.func(bgr_l, bgr_r, num_disparities=64, block_size=15)
    assert out.shape == left.shape


# -------------------------------------------------------------- Stereo SGBM


def test_stereo_sgbm_returns_uint8_singlechannel() -> None:
    left, right = _stereo_pair()
    out = STEREO_SGBM.func(
        left, right,
        num_disparities=64,
        block_size=5,
        min_disparity=0,
        uniqueness_ratio=10,
        speckle_window_size=100,
    )
    assert out.dtype == np.uint8
    assert out.ndim == 2
    assert out.shape == left.shape


def test_stereo_sgbm_resizes_right_to_left() -> None:
    left, right = _stereo_pair()
    smaller_right = cv2.resize(right, (right.shape[1] // 2, right.shape[0] // 2))
    out = STEREO_SGBM.func(
        left, smaller_right,
        num_disparities=64,
        block_size=5,
        min_disparity=0,
        uniqueness_ratio=10,
        speckle_window_size=100,
    )
    assert out.shape == left.shape


def test_stereo_sgbm_produces_non_uniform_output_on_textured_pair() -> None:
    left, right = _stereo_pair()
    out = STEREO_SGBM.func(
        left, right,
        num_disparities=64,
        block_size=5,
        min_disparity=0,
        uniqueness_ratio=10,
        speckle_window_size=100,
    )
    # If matching worked at all, the disparity image will not be a single value.
    assert int(out.std()) > 0
