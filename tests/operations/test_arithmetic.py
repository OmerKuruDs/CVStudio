from __future__ import annotations

import numpy as np

from cvstudio.operations.arithmetic import (
    ADD,
    BITWISE_AND,
    BITWISE_OR,
    BITWISE_XOR,
    MULTIPLY,
    SUBTRACT,
)


def _gray(value: int, size: int = 16) -> np.ndarray:
    return np.full((size, size), value, dtype=np.uint8)


def _bgr(b: int, g: int, r: int, size: int = 16) -> np.ndarray:
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[..., 0] = b
    img[..., 1] = g
    img[..., 2] = r
    return img


# --------------------------------------------------------------------- Add


def test_add_saturates_at_255() -> None:
    a = _gray(200)
    b = _gray(100)
    out = ADD.func(a, b)
    assert int(out[0, 0]) == 255  # 200 + 100 would be 300, clipped to 255


def test_add_handles_gray_plus_bgr_via_match() -> None:
    a = _bgr(10, 20, 30)
    b = _gray(5)
    out = ADD.func(a, b)
    # The gray input is broadcast to BGR by coerce_to_match.
    assert out.shape == a.shape
    assert tuple(int(v) for v in out[0, 0]) == (15, 25, 35)


# ----------------------------------------------------------------- Subtract


def test_subtract_saturates_at_zero() -> None:
    a = _gray(50)
    b = _gray(100)
    out = SUBTRACT.func(a, b)
    assert int(out[0, 0]) == 0  # 50 - 100 = -50, clipped to 0


def test_subtract_preserves_positive_difference() -> None:
    a = _gray(150)
    b = _gray(40)
    out = SUBTRACT.func(a, b)
    assert int(out[0, 0]) == 110


# ----------------------------------------------------------------- Multiply


def test_multiply_with_mask_keeps_image_in_range() -> None:
    img = _gray(200)
    mask = _gray(255)  # full white "mask"
    out = MULTIPLY.func(img, mask, scale=1.0 / 255.0)
    # 200 * 255 * (1/255) = 200
    assert abs(int(out[0, 0]) - 200) <= 1


def test_multiply_with_zero_mask_blanks_image() -> None:
    img = _gray(200)
    mask = _gray(0)
    out = MULTIPLY.func(img, mask, scale=1.0 / 255.0)
    assert int(out[0, 0]) == 0


# ------------------------------------------------------------- Bitwise ops


def test_bitwise_and_is_pixelwise() -> None:
    a = _gray(0b11110000)
    b = _gray(0b10101010)
    out = BITWISE_AND.func(a, b)
    assert int(out[0, 0]) == 0b10100000


def test_bitwise_or_is_pixelwise() -> None:
    a = _gray(0b00110011)
    b = _gray(0b10101010)
    out = BITWISE_OR.func(a, b)
    assert int(out[0, 0]) == 0b10111011


def test_bitwise_xor_is_pixelwise() -> None:
    a = _gray(0b11110000)
    b = _gray(0b10101010)
    out = BITWISE_XOR.func(a, b)
    assert int(out[0, 0]) == 0b01011010


def test_bitwise_xor_with_self_is_zero() -> None:
    a = _gray(123)
    out = BITWISE_XOR.func(a, a)
    assert int(out.max()) == 0
