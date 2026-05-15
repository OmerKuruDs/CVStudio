from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from cvstudio.core.image_io import read_image, to_uint8


def test_to_uint8_passes_through_uint8_arrays() -> None:
    img = np.full((4, 4, 3), 50, dtype=np.uint8)
    out = to_uint8(img)
    assert out is img  # zero-copy when no work needed


def test_to_uint8_rescales_uint16_to_high_byte() -> None:
    img = np.array([[0, 256, 32_768, 65_280, 65_535]], dtype=np.uint16)
    out = to_uint8(img)
    # `image >> 8` keeps the top byte; the boundary values round predictably.
    assert out.dtype == np.uint8
    assert out.tolist() == [[0, 1, 128, 255, 255]]


def test_to_uint8_normalises_float_to_full_range() -> None:
    img = np.array([[0.2, 0.4, 0.6, 0.8]], dtype=np.float32)
    out = to_uint8(img)
    assert out.dtype == np.uint8
    # Min=0.2 maps to 0, max=0.8 maps to 255; midpoints scale linearly.
    assert out[0, 0] == 0
    assert out[0, -1] == 255
    # The values in between should be monotonically increasing.
    assert (np.diff(out[0]) >= 0).all()


def test_to_uint8_handles_constant_float_image_as_zero() -> None:
    img = np.full((3, 3), 0.42, dtype=np.float32)
    out = to_uint8(img)
    assert out.dtype == np.uint8
    assert (out == 0).all()


def test_read_image_returns_none_for_missing_file(tmp_path: Path) -> None:
    assert read_image(tmp_path / "does_not_exist.tif") is None


def test_read_image_round_trips_uint8_png(tmp_path: Path) -> None:
    src = np.full((4, 4, 3), 120, dtype=np.uint8)
    target = tmp_path / "image.png"
    cv2.imwrite(str(target), src)
    out = read_image(target)
    assert out is not None
    assert out.dtype == np.uint8
    assert int(out[0, 0, 0]) == 120


def test_read_image_normalises_16bit_tiff(tmp_path: Path) -> None:
    """A 16-bit TIFF used to load but display all-white because the pipeline
    naively clipped to 255. After this fix it round-trips into a visible 8-bit
    image whose dynamic range matches the upper byte of the source."""
    src = np.array(
        [[0, 65_535, 0], [65_535, 32_768, 0], [0, 0, 65_535]], dtype=np.uint16
    )
    target = tmp_path / "image.tif"
    cv2.imwrite(str(target), src)
    out = read_image(target)
    assert out is not None
    assert out.dtype == np.uint8
    # Bright pixels survive, dark pixels stay dark.
    assert out[0, 1] == 255
    assert out[1, 1] in (127, 128)  # 32768 >> 8
    assert out[0, 0] == 0


def test_read_thumbnail_returns_uint8_within_max_dim(tmp_path: Path) -> None:
    from cvstudio.core.image_io import read_thumbnail

    big = np.full((800, 1200, 3), 50, dtype=np.uint8)
    target = tmp_path / "big.png"
    cv2.imwrite(str(target), big)
    thumb = read_thumbnail(target, max_dim=200)
    assert thumb is not None
    assert thumb.dtype == np.uint8
    h, w = thumb.shape[:2]
    assert max(h, w) <= 200


def test_read_thumbnail_returns_none_for_missing_file(tmp_path: Path) -> None:
    from cvstudio.core.image_io import read_thumbnail

    assert read_thumbnail(tmp_path / "does_not_exist.png") is None


def test_read_image_normalises_float_tiff(tmp_path: Path) -> None:
    """OpenCV writes float32 TIFFs when given float arrays. The loader should
    rescale them into the visible 0..255 band rather than clipping to white."""
    src = np.array([[0.1, 0.5], [0.9, 0.3]], dtype=np.float32)
    target = tmp_path / "float.tif"
    cv2.imwrite(str(target), src)
    out = read_image(target)
    assert out is not None
    assert out.dtype == np.uint8
    # Min input pixel → 0, max input pixel → 255.
    assert int(out.min()) == 0
    assert int(out.max()) == 255
