from __future__ import annotations

import numpy as np

from cvstudio.operations.edge import CANNY, LAPLACIAN, SCHARR, SOBEL


def _vertical_step() -> np.ndarray:
    """Half-black, half-white image with one strong vertical edge in the middle."""
    img = np.zeros((32, 32), dtype=np.uint8)
    img[:, 16:] = 255
    return img


def test_canny_finds_the_vertical_edge() -> None:
    out = CANNY.func(_vertical_step(), threshold1=100, threshold2=200, aperture_size=3)
    # An edge pixel should sit on the boundary column (15 or 16).
    edge_columns = np.where(out.any(axis=0))[0]
    assert any(col in {15, 16} for col in edge_columns)
    assert out.dtype == np.uint8


def test_canny_accepts_bgr_input() -> None:
    bgr = np.stack([_vertical_step()] * 3, axis=-1)
    out = CANNY.func(bgr, threshold1=100, threshold2=200, aperture_size=3)
    assert out.ndim == 2


def test_sobel_dx_responds_to_vertical_edge() -> None:
    out = SOBEL.func(_vertical_step(), dx=1, dy=0, ksize=3)
    # Gradient should be strongest near the edge column.
    column_sums = out.sum(axis=0).astype(np.int64)
    peak_col = int(np.argmax(column_sums))
    assert peak_col in {14, 15, 16, 17}


def test_sobel_zero_orders_pass_through_grayscale() -> None:
    img = np.full((4, 4, 3), 100, dtype=np.uint8)
    out = SOBEL.func(img, dx=0, dy=0, ksize=3)
    # Grayscale of a flat color should be a single value, no derivative applied.
    assert out.ndim == 2
    assert np.all(out == out[0, 0])


def test_laplacian_returns_uint8_edge_map() -> None:
    out = LAPLACIAN.func(_vertical_step(), ksize=3)
    assert out.dtype == np.uint8
    assert out.max() > 0  # the edge should produce some response


def test_scharr_dx_responds_to_vertical_edge() -> None:
    out = SCHARR.func(_vertical_step(), dx=1, dy=0)
    column_sums = out.sum(axis=0).astype(np.int64)
    peak_col = int(np.argmax(column_sums))
    assert peak_col in {14, 15, 16, 17}
    assert out.dtype == np.uint8


def test_scharr_zero_orders_pass_through_grayscale() -> None:
    img = np.full((4, 4, 3), 100, dtype=np.uint8)
    out = SCHARR.func(img, dx=0, dy=0)
    assert out.ndim == 2
    assert np.all(out == out[0, 0])


def test_scharr_magnitude_mode_combines_directions() -> None:
    img = _vertical_step()
    only_x = SCHARR.func(img, dx=1, dy=0)
    magnitude = SCHARR.func(img, dx=1, dy=1)
    # The dx=dy=1 magnitude composite must be at least as strong as dx alone.
    assert int(magnitude.max()) >= int(only_x.max())
