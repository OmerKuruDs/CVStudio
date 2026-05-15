from __future__ import annotations

import numpy as np

from cvstudio.ui.main_window import PREVIEW_MAX_DIM, downscale_for_preview


def test_small_image_is_returned_unchanged() -> None:
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    out = downscale_for_preview(img)
    assert out is img  # identity preserved — no allocation when no work to do


def test_image_at_exactly_the_cap_is_unchanged() -> None:
    img = np.zeros((PREVIEW_MAX_DIM, PREVIEW_MAX_DIM // 2, 3), dtype=np.uint8)
    out = downscale_for_preview(img)
    assert out is img


def test_landscape_image_is_capped_on_width() -> None:
    img = np.zeros((1200, 3200, 3), dtype=np.uint8)  # w > h, w > cap
    out = downscale_for_preview(img)
    h, w = out.shape[:2]
    assert w == PREVIEW_MAX_DIM
    # Aspect ratio preserved within 1-pixel rounding.
    expected_h = round(1200 * (PREVIEW_MAX_DIM / 3200))
    assert h == expected_h


def test_portrait_image_is_capped_on_height() -> None:
    img = np.zeros((3000, 1200, 3), dtype=np.uint8)  # h > w, h > cap
    out = downscale_for_preview(img)
    h, w = out.shape[:2]
    assert h == PREVIEW_MAX_DIM
    expected_w = round(1200 * (PREVIEW_MAX_DIM / 3000))
    assert w == expected_w


def test_square_image_is_capped_on_both_axes() -> None:
    img = np.zeros((4000, 4000, 3), dtype=np.uint8)
    out = downscale_for_preview(img)
    assert out.shape[:2] == (PREVIEW_MAX_DIM, PREVIEW_MAX_DIM)


def test_grayscale_image_keeps_single_channel() -> None:
    img = np.zeros((3000, 3000), dtype=np.uint8)
    out = downscale_for_preview(img)
    assert out.ndim == 2
    assert out.shape == (PREVIEW_MAX_DIM, PREVIEW_MAX_DIM)


def test_custom_max_dim_is_respected() -> None:
    img = np.zeros((1000, 1000, 3), dtype=np.uint8)
    out = downscale_for_preview(img, max_dim=400)
    assert out.shape[:2] == (400, 400)


def test_preserves_dtype() -> None:
    img = np.zeros((3000, 3000, 3), dtype=np.uint8)
    out = downscale_for_preview(img)
    assert out.dtype == np.uint8
