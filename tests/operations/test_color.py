from __future__ import annotations

import numpy as np
import pytest

from cvstudio.operations.color import (
    CHANNEL,
    CLAHE,
    HSV_IN_RANGE,
    INVERT,
    TO_GRAYSCALE,
    TO_HSV,
)


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


def test_clahe_on_grayscale_returns_single_channel() -> None:
    gray = np.random.default_rng(0).integers(0, 255, size=(16, 16), dtype=np.uint8)
    out = CLAHE.func(gray, clip_limit=2.0, tile_grid=4)
    assert out.shape == gray.shape
    assert out.dtype == np.uint8


def test_clahe_on_color_returns_bgr() -> None:
    img = np.random.default_rng(0).integers(0, 255, size=(16, 16, 3), dtype=np.uint8)
    out = CLAHE.func(img, clip_limit=2.0, tile_grid=4)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_clahe_actually_stretches_low_contrast_grayscale() -> None:
    # Low-contrast input: values clustered tightly around 128.
    gray = np.full((32, 32), 128, dtype=np.uint8)
    gray[::2, ::2] = 130
    gray[1::2, 1::2] = 126
    out = CLAHE.func(gray, clip_limit=4.0, tile_grid=8)
    assert out.max() - out.min() > gray.max() - gray.min()


def test_hsv_in_range_picks_red() -> None:
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    img[..., 2] = 255  # pure red in BGR
    mask = HSV_IN_RANGE.func(
        img, h_min=0, h_max=10, s_min=200, s_max=255, v_min=100, v_max=255
    )
    assert mask.ndim == 2
    assert mask.dtype == np.uint8
    assert (mask == 255).all()


def test_hsv_in_range_rejects_grayscale_input() -> None:
    gray = np.zeros((4, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="3-channel"):
        HSV_IN_RANGE.func(
            gray, h_min=0, h_max=10, s_min=0, s_max=255, v_min=0, v_max=255
        )
