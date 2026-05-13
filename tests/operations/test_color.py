from __future__ import annotations

import numpy as np
import pytest

from cvsandbox.operations.color import CHANNEL, INVERT, TO_GRAYSCALE, TO_HSV


def _bgr_image() -> np.ndarray:
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    img[..., 0] = 50  # B
    img[..., 1] = 100  # G
    img[..., 2] = 150  # R
    return img


def test_to_grayscale_returns_single_channel() -> None:
    out = TO_GRAYSCALE.func(_bgr_image())
    assert out.ndim == 2


def test_to_grayscale_is_a_passthrough_for_gray_input() -> None:
    gray = np.full((4, 4), 100, dtype=np.uint8)
    out = TO_GRAYSCALE.func(gray)
    assert out.shape == gray.shape
    assert np.array_equal(out, gray)


def test_to_hsv_requires_three_channels() -> None:
    with pytest.raises(ValueError, match="3-channel"):
        TO_HSV.func(np.zeros((4, 4), dtype=np.uint8))


def test_invert_negates_pixels() -> None:
    out = INVERT.func(np.full((2, 2), 30, dtype=np.uint8))
    assert int(out[0, 0]) == 225  # 255 - 30


def test_channel_extracts_blue() -> None:
    out = CHANNEL.func(_bgr_image(), channel=0)
    assert out.ndim == 2
    assert int(out[0, 0]) == 50


def test_channel_index_wraps_within_image_channels() -> None:
    # Channel 2 on a 3-channel image picks the red channel.
    out = CHANNEL.func(_bgr_image(), channel=2)
    assert int(out[0, 0]) == 150
