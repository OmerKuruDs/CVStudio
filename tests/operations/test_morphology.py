from __future__ import annotations

import numpy as np

from cvstudio.operations.morphology import (
    BLACKHAT,
    CLOSE,
    DILATE,
    ERODE,
    GRADIENT,
    OPEN,
    TOPHAT,
)


def _dot_image() -> np.ndarray:
    img = np.zeros((11, 11), dtype=np.uint8)
    img[5, 5] = 255
    return img


def _ring_image() -> np.ndarray:
    """Bright square with a small dark hole — ideal for testing 'close'."""
    img = np.full((11, 11), 255, dtype=np.uint8)
    img[5, 5] = 0
    return img


def test_erode_shrinks_bright_region() -> None:
    img = np.full((11, 11), 255, dtype=np.uint8)
    img[0, :] = 0
    out = ERODE.func(img, shape="Rectangle", ksize=3, iterations=1)
    # The top two rows should now be dark (one ate into row 1).
    assert out[1, 5] == 0


def test_dilate_grows_bright_pixel() -> None:
    out = DILATE.func(_dot_image(), shape="Rectangle", ksize=3, iterations=1)
    assert out[4, 5] == 255
    assert out[5, 4] == 255


def test_open_removes_isolated_bright_pixel() -> None:
    out = OPEN.func(_dot_image(), shape="Rectangle", ksize=3, iterations=1)
    assert out[5, 5] == 0


def test_close_fills_isolated_dark_hole() -> None:
    out = CLOSE.func(_ring_image(), shape="Rectangle", ksize=3, iterations=1)
    assert out[5, 5] == 255


def test_gradient_outlines_bright_square() -> None:
    img = np.zeros((11, 11), dtype=np.uint8)
    img[3:8, 3:8] = 255
    out = GRADIENT.func(img, shape="Rectangle", ksize=3, iterations=1)
    # Outline pixels should be bright; the inside should be dark again.
    assert out[5, 5] == 0  # interior cancels
    assert out[3, 3] > 0  # rim survives


def test_tophat_isolates_small_bright_feature() -> None:
    img = np.zeros((21, 21), dtype=np.uint8)
    img[10, 10] = 255
    out = TOPHAT.func(img, shape="Rectangle", ksize=5, iterations=1)
    # Top-Hat (input - opening) should preserve the isolated bright pixel.
    assert out[10, 10] == 255


def test_blackhat_isolates_small_dark_feature() -> None:
    img = np.full((21, 21), 255, dtype=np.uint8)
    img[10, 10] = 0
    out = BLACKHAT.func(img, shape="Rectangle", ksize=5, iterations=1)
    # Black-Hat (closing - input) should yield a bright spot at the hole.
    assert out[10, 10] == 255
