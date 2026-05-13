from __future__ import annotations

import numpy as np

from cvsandbox.operations.geometric import FLIP, RESIZE, ROTATE


def test_resize_scales_both_axes() -> None:
    img = np.zeros((10, 20, 3), dtype=np.uint8)
    out = RESIZE.func(img, scale_x=0.5, scale_y=0.5, interpolation="Linear")
    assert out.shape == (5, 10, 3)


def test_resize_independent_axis_factors() -> None:
    img = np.zeros((10, 20), dtype=np.uint8)
    out = RESIZE.func(img, scale_x=2.0, scale_y=0.5, interpolation="Nearest")
    assert out.shape == (5, 40)


def test_rotate_360_returns_visually_same_image() -> None:
    img = np.random.default_rng(0).integers(0, 255, size=(16, 16), dtype=np.uint8)
    out = ROTATE.func(img, angle=360.0)
    # warpAffine round-trips can leave 1-pixel artifacts on the border, so we
    # compare the interior only.
    np.testing.assert_array_equal(out[2:-2, 2:-2], img[2:-2, 2:-2])


def test_rotate_preserves_shape() -> None:
    img = np.zeros((10, 20, 3), dtype=np.uint8)
    out = ROTATE.func(img, angle=45.0)
    assert out.shape == img.shape


def test_flip_horizontal_swaps_columns() -> None:
    img = np.zeros((4, 4), dtype=np.uint8)
    img[:, 0] = 255
    out = FLIP.func(img, mode="Horizontal")
    assert int(out[0, 0]) == 0
    assert int(out[0, -1]) == 255


def test_flip_vertical_swaps_rows() -> None:
    img = np.zeros((4, 4), dtype=np.uint8)
    img[0, :] = 255
    out = FLIP.func(img, mode="Vertical")
    assert int(out[0, 0]) == 0
    assert int(out[-1, 0]) == 255
