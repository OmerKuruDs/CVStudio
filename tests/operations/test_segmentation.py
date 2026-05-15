from __future__ import annotations

import cv2
import numpy as np

from cvstudio.operations.segmentation import (
    CONNECTED_COMPONENTS,
    DISTANCE_TRANSFORM,
    GRABCUT,
    WATERSHED,
)


def _three_blob_mask() -> np.ndarray:
    mask = np.zeros((80, 80), dtype=np.uint8)
    mask[10:20, 10:20] = 255  # 10x10 = 100 px
    mask[30:50, 30:50] = 255  # 20x20 = 400 px
    mask[60:62, 60:62] = 255  # 2x2 = 4 px (small)
    return mask


def _bright_object_image() -> np.ndarray:
    """A solid white circle on dark grey background — single blob for watershed."""
    img = np.full((80, 80, 3), 30, dtype=np.uint8)
    cv2.circle(img, (40, 40), 18, (220, 220, 220), thickness=-1)
    return img


def _foreground_subject() -> np.ndarray:
    """An obvious foreground rectangle near the centre for GrabCut to pick out."""
    img = np.full((100, 100, 3), 20, dtype=np.uint8)
    cv2.rectangle(img, (30, 30), (70, 70), (200, 150, 100), thickness=-1)
    return img


# ----------------------------------------------------------- Distance Transform


def test_distance_transform_grayscale_output() -> None:
    out = DISTANCE_TRANSFORM.func(
        _three_blob_mask(),
        distance_type="L2 (Euclidean)",
        mask_size="3",
        color_map=False,
    )
    assert out.ndim == 2
    assert out.dtype == np.uint8
    # The centre of the big blob should be brighter than its rim.
    assert int(out[40, 40]) > int(out[31, 31])


def test_distance_transform_color_output() -> None:
    out = DISTANCE_TRANSFORM.func(
        _three_blob_mask(),
        distance_type="L2 (Euclidean)",
        mask_size="3",
        color_map=True,
    )
    assert out.ndim == 3
    assert out.shape[2] == 3


def test_distance_transform_accepts_bgr_input() -> None:
    bgr = cv2.cvtColor(_three_blob_mask(), cv2.COLOR_GRAY2BGR)
    out = DISTANCE_TRANSFORM.func(
        bgr,
        distance_type="L1 (Manhattan)",
        mask_size="3",
        color_map=False,
    )
    assert out.shape == (80, 80)


# ----------------------------------------------------------- Connected Components


def test_connected_components_assigns_distinct_colors() -> None:
    out = CONNECTED_COMPONENTS.func(_three_blob_mask(), connectivity="8-way", min_area=0)
    assert out.ndim == 3
    assert out.shape[2] == 3
    # Three blobs should yield ≥ 3 distinct non-black colours in the output.
    colors = {tuple(c) for c in out.reshape(-1, 3) if not (c[0] == 0 and c[1] == 0 and c[2] == 0)}
    assert len(colors) >= 3


def test_connected_components_filters_small_blob() -> None:
    full = CONNECTED_COMPONENTS.func(_three_blob_mask(), connectivity="8-way", min_area=0)
    big_only = CONNECTED_COMPONENTS.func(
        _three_blob_mask(), connectivity="8-way", min_area=50
    )
    full_nonblack = ((full != 0).any(axis=-1)).sum()
    big_nonblack = ((big_only != 0).any(axis=-1)).sum()
    # The 4-pixel blob should be dropped → fewer painted pixels.
    assert big_nonblack < full_nonblack


def test_connected_components_background_stays_black() -> None:
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:10, 5:10] = 255
    out = CONNECTED_COMPONENTS.func(mask, connectivity="8-way", min_area=0)
    # The background pixel at (0, 0) must be exactly black.
    assert tuple(out[0, 0]) == (0, 0, 0)


# ---------------------------------------------------------------------- Watershed


def test_watershed_draws_red_boundary() -> None:
    out = WATERSHED.func(
        _bright_object_image(),
        foreground_threshold=0.5,
        bg_dilate_iters=3,
        noise_kernel=3,
    )
    assert out.shape == (80, 80, 3)
    is_red = (out[..., 0] == 0) & (out[..., 1] == 0) & (out[..., 2] == 255)
    assert is_red.any()


def test_watershed_runs_on_grayscale_input() -> None:
    gray = cv2.cvtColor(_bright_object_image(), cv2.COLOR_BGR2GRAY)
    out = WATERSHED.func(
        gray,
        foreground_threshold=0.5,
        bg_dilate_iters=3,
        noise_kernel=3,
    )
    assert out.ndim == 3 and out.shape[2] == 3


# ----------------------------------------------------------------------- GrabCut


def test_grabcut_extracts_central_foreground() -> None:
    img = _foreground_subject()
    out = GRABCUT.func(img, margin_pct=15, iterations=3)
    assert out.shape == img.shape
    # The corner (clearly outside the rect) must be black.
    assert tuple(out[0, 0]) == (0, 0, 0)
    # The centre (subject) must be roughly preserved.
    assert tuple(out[50, 50]) != (0, 0, 0)


def test_grabcut_returns_bgr_for_grayscale_input() -> None:
    img = np.full((60, 60), 100, dtype=np.uint8)
    img[20:40, 20:40] = 220
    out = GRABCUT.func(img, margin_pct=10, iterations=2)
    assert out.ndim == 3 and out.shape[2] == 3
