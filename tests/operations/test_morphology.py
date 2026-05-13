from __future__ import annotations

import numpy as np

from cvsandbox.operations.morphology import CLOSE, DILATE, ERODE, OPEN


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
