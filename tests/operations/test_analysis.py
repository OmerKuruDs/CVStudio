from __future__ import annotations

import numpy as np

from cvstudio.operations.analysis import FIND_CONTOURS


def _mask_with_two_blobs() -> np.ndarray:
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[4:10, 4:10] = 255  # 6x6 = 36 px
    mask[20:30, 20:30] = 255  # 10x10 = 100 px
    return mask


def test_find_contours_returns_bgr_canvas_for_gray_input() -> None:
    out = FIND_CONTOURS.func(_mask_with_two_blobs(), mode="External", min_area=0, thickness=2)
    assert out.ndim == 3
    assert out.shape[2] == 3
    assert out.dtype == np.uint8


def test_find_contours_draws_green() -> None:
    mask = _mask_with_two_blobs()
    out = FIND_CONTOURS.func(mask, mode="External", min_area=0, thickness=2)
    # At least one green pixel should be drawn somewhere on the canvas.
    is_green = (out[..., 0] == 0) & (out[..., 1] == 255) & (out[..., 2] == 0)
    assert is_green.any()


def test_find_contours_min_area_filters_small_blob() -> None:
    mask = _mask_with_two_blobs()  # blobs of area 36 and 100
    out_all = FIND_CONTOURS.func(mask, mode="External", min_area=0, thickness=-1)
    out_big = FIND_CONTOURS.func(mask, mode="External", min_area=50, thickness=-1)
    green_all = ((out_all[..., 0] == 0) & (out_all[..., 1] == 255) & (out_all[..., 2] == 0)).sum()
    green_big = ((out_big[..., 0] == 0) & (out_big[..., 1] == 255) & (out_big[..., 2] == 0)).sum()
    assert green_big < green_all  # small blob is gone


def test_find_contours_accepts_bgr_input() -> None:
    bgr = np.zeros((32, 32, 3), dtype=np.uint8)
    bgr[4:10, 4:10, :] = 255  # white square
    out = FIND_CONTOURS.func(bgr, mode="External", min_area=0, thickness=1)
    assert out.shape == bgr.shape
