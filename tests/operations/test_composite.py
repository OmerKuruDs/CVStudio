from __future__ import annotations

import numpy as np

from cvstudio.operations.composite import APPLY_MASK, BLEND, DIFFERENCE


def test_blend_default_alpha_is_a_midpoint_average() -> None:
    a = np.full((4, 4, 3), 80, dtype=np.uint8)
    b = np.full((4, 4, 3), 200, dtype=np.uint8)
    out = BLEND.func(a, b, alpha=0.5)
    assert int(out[0, 0, 0]) == 140  # (80 + 200) / 2


def test_blend_alpha_zero_returns_a_only() -> None:
    a = np.full((4, 4, 3), 50, dtype=np.uint8)
    b = np.full((4, 4, 3), 200, dtype=np.uint8)
    out = BLEND.func(a, b, alpha=0.0)
    assert int(out[0, 0, 0]) == 50


def test_blend_promotes_grayscale_b_to_match_a() -> None:
    a = np.full((4, 4, 3), 100, dtype=np.uint8)
    b_gray = np.full((4, 4), 200, dtype=np.uint8)
    out = BLEND.func(a, b_gray, alpha=0.5)
    assert out.shape == a.shape
    assert int(out[0, 0, 0]) == 150


def test_blend_resizes_b_to_match_a() -> None:
    a = np.full((6, 6, 3), 100, dtype=np.uint8)
    b = np.full((3, 3, 3), 200, dtype=np.uint8)
    out = BLEND.func(a, b, alpha=0.5)
    assert out.shape == a.shape


def test_apply_mask_zeros_pixels_where_mask_is_zero() -> None:
    img = np.full((4, 4, 3), 200, dtype=np.uint8)
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 255
    out = APPLY_MASK.func(img, mask)
    assert int(out[0, 0, 0]) == 0
    assert int(out[2, 2, 0]) == 200


def test_apply_mask_accepts_color_mask() -> None:
    img = np.full((4, 4, 3), 50, dtype=np.uint8)
    color_mask = np.zeros((4, 4, 3), dtype=np.uint8)
    color_mask[..., 0] = 100  # blue channel non-zero
    out = APPLY_MASK.func(img, color_mask)
    # cvtColor(BGR2GRAY) keeps non-zero pixels — every pixel becomes a keep pixel.
    assert (out == img).all()


def test_difference_returns_abs_pixel_delta() -> None:
    a = np.full((4, 4, 3), 100, dtype=np.uint8)
    b = np.full((4, 4, 3), 130, dtype=np.uint8)
    out = DIFFERENCE.func(a, b)
    assert int(out[0, 0, 0]) == 30


def test_difference_is_symmetric() -> None:
    a = np.full((4, 4, 3), 100, dtype=np.uint8)
    b = np.full((4, 4, 3), 200, dtype=np.uint8)
    assert int(DIFFERENCE.func(a, b)[0, 0, 0]) == int(DIFFERENCE.func(b, a)[0, 0, 0])


def test_composite_specs_declare_two_input_ports() -> None:
    assert BLEND.input_ports == ("a", "b")
    assert APPLY_MASK.input_ports == ("image", "mask")
    assert DIFFERENCE.input_ports == ("a", "b")
