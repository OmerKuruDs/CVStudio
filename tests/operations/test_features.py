from __future__ import annotations

import numpy as np

from cvstudio.operations.features import (
    FAST,
    HARRIS,
    HOUGH_CIRCLES,
    HOUGH_LINES,
    ORB,
    SHI_TOMASI,
)


def _checkerboard(size: int = 64, square: int = 16) -> np.ndarray:
    """Grayscale checkerboard — produces strong, predictable Harris corners."""
    img = np.zeros((size, size), dtype=np.uint8)
    for y in range(0, size, square):
        for x in range(0, size, square):
            if ((x // square) + (y // square)) % 2 == 0:
                img[y : y + square, x : x + square] = 255
    return img


def _textured_image(seed: int = 0, size: int = 120) -> np.ndarray:
    """Random shapes on a noisy background — FAST/ORB find many keypoints."""
    import cv2

    rng = np.random.default_rng(seed)
    img = rng.integers(40, 80, size=(size, size), dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (40, 60), 220, thickness=-1)
    cv2.rectangle(img, (60, 30), (100, 80), 200, thickness=2)
    cv2.circle(img, (80, 20), 12, 255, thickness=2)
    cv2.line(img, (5, 100), (100, 110), 240, thickness=2)
    return img


def _circle_image(size: int = 200, radius: int = 40) -> np.ndarray:
    img = np.zeros((size, size), dtype=np.uint8)
    import cv2

    cv2.circle(img, (size // 2, size // 2), radius, 255, thickness=2)
    return img


def _line_image() -> np.ndarray:
    """A 200x200 image with a single bright diagonal-ish line."""
    img = np.zeros((200, 200), dtype=np.uint8)
    import cv2

    cv2.line(img, (10, 10), (190, 60), 255, thickness=2)
    return img


# ---------------------------------------------------------------- Harris


def test_harris_returns_bgr_with_red_corners() -> None:
    out = HARRIS.func(_checkerboard(), block_size=2, ksize=3, k=0.04, threshold=0.01)
    assert out.ndim == 3 and out.shape[2] == 3
    assert out.dtype == np.uint8
    # At least some pure-red pixels (BGR: B=0, G=0, R=255) should be drawn.
    is_red = (out[..., 0] == 0) & (out[..., 1] == 0) & (out[..., 2] == 255)
    assert is_red.any()


def test_harris_accepts_bgr_input() -> None:
    bgr = np.stack([_checkerboard()] * 3, axis=-1)
    out = HARRIS.func(bgr, block_size=2, ksize=3, k=0.04, threshold=0.01)
    assert out.shape == bgr.shape


def test_harris_high_threshold_marks_fewer_pixels() -> None:
    img = _checkerboard()
    low = HARRIS.func(img, block_size=2, ksize=3, k=0.04, threshold=0.01)
    high = HARRIS.func(img, block_size=2, ksize=3, k=0.04, threshold=0.5)
    red_low = ((low[..., 2] == 255) & (low[..., 0] == 0)).sum()
    red_high = ((high[..., 2] == 255) & (high[..., 0] == 0)).sum()
    assert red_high <= red_low


# ---------------------------------------------------------------- Shi-Tomasi


def test_shi_tomasi_draws_green_circles() -> None:
    out = SHI_TOMASI.func(
        _checkerboard(),
        max_corners=20,
        quality_level=0.01,
        min_distance=5,
        block_size=3,
        radius=3,
    )
    is_green = (out[..., 0] == 0) & (out[..., 1] == 255) & (out[..., 2] == 0)
    assert is_green.any()


def test_shi_tomasi_handles_no_corners_gracefully() -> None:
    flat = np.full((40, 40), 128, dtype=np.uint8)
    out = SHI_TOMASI.func(
        flat,
        max_corners=100,
        quality_level=0.5,
        min_distance=10,
        block_size=3,
        radius=3,
    )
    assert out.shape == (40, 40, 3)
    assert out.dtype == np.uint8


# --------------------------------------------------------------------- FAST


def test_fast_marks_corners_on_bgr_canvas() -> None:
    img = _textured_image()
    baseline = np.stack([img] * 3, axis=-1)
    out = FAST.func(img, threshold=10, nonmax=True)
    assert out.ndim == 3 and out.shape[2] == 3
    # drawKeypoints anti-aliases, so check that *something* was overlaid.
    assert not np.array_equal(out, baseline)
    # Yellow = high G + high R, low B. The blue channel should drop somewhere.
    blue_drop = (out[..., 0].astype(int) - baseline[..., 0].astype(int)) < -10
    assert blue_drop.any()


def test_fast_threshold_higher_yields_fewer_or_equal_modified_pixels() -> None:
    img = _textured_image()
    baseline = np.stack([img] * 3, axis=-1)
    low = FAST.func(img, threshold=5, nonmax=True)
    high = FAST.func(img, threshold=80, nonmax=True)
    low_diff = (low != baseline).any(axis=-1).sum()
    high_diff = (high != baseline).any(axis=-1).sum()
    assert high_diff <= low_diff


# ---------------------------------------------------------------------- ORB


def test_orb_draws_magenta_keypoints() -> None:
    img = _textured_image()
    baseline = np.stack([img] * 3, axis=-1)
    out = ORB.func(img, nfeatures=50, scale_factor=1.2, nlevels=4, rich=False)
    assert out.ndim == 3 and out.shape[2] == 3
    # drawKeypoints anti-aliases magenta — assert the canvas was modified.
    assert not np.array_equal(out, baseline)
    # Green channel should drop (magenta = high R + B, low G).
    green_drop = (out[..., 1].astype(int) - baseline[..., 1].astype(int)) < -10
    assert green_drop.any()


def test_orb_rich_keypoints_paints_more_than_plain() -> None:
    img = _textured_image()
    baseline = np.stack([img] * 3, axis=-1)
    plain = ORB.func(img, nfeatures=50, scale_factor=1.2, nlevels=4, rich=False)
    rich = ORB.func(img, nfeatures=50, scale_factor=1.2, nlevels=4, rich=True)
    plain_diff = (plain != baseline).any(axis=-1).sum()
    rich_diff = (rich != baseline).any(axis=-1).sum()
    # Rich draws orientation circles + radial line → more modified pixels.
    assert rich_diff >= plain_diff


# --------------------------------------------------------------- Hough Lines


def test_hough_lines_detects_the_diagonal() -> None:
    out = HOUGH_LINES.func(
        _line_image(),
        canny_low=50,
        canny_high=150,
        threshold=30,
        min_line_length=20,
        max_line_gap=5,
        thickness=2,
    )
    is_green = (out[..., 0] == 0) & (out[..., 1] == 255) & (out[..., 2] == 0)
    assert is_green.any()


def test_hough_lines_returns_unchanged_bgr_when_no_line() -> None:
    flat = np.full((40, 40), 50, dtype=np.uint8)
    out = HOUGH_LINES.func(
        flat,
        canny_low=100,
        canny_high=200,
        threshold=80,
        min_line_length=50,
        max_line_gap=10,
        thickness=2,
    )
    assert out.shape == (40, 40, 3)
    # No green should be drawn since there are no edges.
    is_green = (out[..., 0] == 0) & (out[..., 1] == 255) & (out[..., 2] == 0)
    assert not is_green.any()


# ------------------------------------------------------------- Hough Circles


def test_hough_circles_detects_a_drawn_circle() -> None:
    out = HOUGH_CIRCLES.func(
        _circle_image(),
        dp=1.5,
        min_dist=30,
        canny_threshold=100,
        accumulator_threshold=20,
        min_radius=20,
        max_radius=60,
        thickness=2,
    )
    assert out.ndim == 3 and out.shape[2] == 3
    # Rim is green, center dot is red — at least one of each.
    is_green = (out[..., 0] == 0) & (out[..., 1] == 255) & (out[..., 2] == 0)
    is_red = (out[..., 0] == 0) & (out[..., 1] == 0) & (out[..., 2] == 255)
    assert is_green.any() and is_red.any()


def test_hough_circles_no_detection_keeps_canvas() -> None:
    flat = np.full((40, 40), 50, dtype=np.uint8)
    out = HOUGH_CIRCLES.func(
        flat,
        dp=1.5,
        min_dist=30,
        canny_threshold=100,
        accumulator_threshold=30,
        min_radius=5,
        max_radius=15,
        thickness=2,
    )
    is_green = (out[..., 0] == 0) & (out[..., 1] == 255) & (out[..., 2] == 0)
    assert not is_green.any()
