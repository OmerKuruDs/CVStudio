"""Shared helpers for grain-deformation defect injectors.

Both ``grain_flattening`` (squeeze along an axis) and ``grain_stretching``
(its inverse) share the same input validation, ROI sampling, ROI clamping,
intensity → ratio mapping, and feathered alpha blend. Centralising them
here keeps the two op implementations honest — any rule change (a new
intensity curve, a tighter ROI clamp, a richer Canny sampler) automatically
applies to both.
"""

from __future__ import annotations

import cv2
import numpy as np

_DEFORM_SLOPE = 0.7
"""intensity 1.0 → ratio 0.30. ratio < 1 narrows (squeeze) or pulls a
narrower source crop (stretch). 0.7 keeps both ops in a regime where the
LANCZOS4 resample stays well-behaved at the extreme."""

_NO_OP_RATIO_EPSILON = 1e-3


def intensity_to_deform_ratio(intensity: float) -> float:
    """intensity ∈ [0, 1] → deformation ratio ∈ [0.30, 1.00]. The same
    mapping for both squeeze (source crop wider than ROI, resized down to
    ratio*w) and stretch (source crop narrower than ROI, resized up from
    ratio*w). Linear: ``1.0 - intensity * 0.7``."""
    clipped = max(0.0, min(1.0, float(intensity)))
    return 1.0 - clipped * _DEFORM_SLOPE


def feathered_alpha(height: int, width: int, fraction: float) -> np.ndarray:
    """Centre-bright, edge-dark alpha mask. The inner rectangle is fully
    opaque (1.0); a Gaussian blur softens the transition to the ROI edge
    so the blended patch fades into the surrounding texture instead of
    leaving a boxy seam. ``fraction`` ∈ [0, 0.5] sets how much of each
    side gets feathered (0.3 = inner 40% solid, outer 30% on each side
    fading)."""
    fraction = max(0.0, min(0.5, float(fraction)))
    margin_y = max(1, int(round(height * fraction)))
    margin_x = max(1, int(round(width * fraction)))
    inner = np.zeros((height, width), dtype=np.float32)
    inner[margin_y : height - margin_y, margin_x : width - margin_x] = 1.0
    blur_size = max(margin_x, margin_y) | 1
    blur_size = max(3, blur_size)
    return cv2.GaussianBlur(inner, (blur_size, blur_size), 0)


def pick_random_roi_near_edge(
    image: np.ndarray, roi_size: int, rng: np.random.Generator
) -> tuple[int, int, int, int]:
    """Sample a Canny edge pixel and centre a square ROI on it so the
    injected defect lands on visible grain structure. Falls back to the
    image centre when Canny finds no edges (e.g. a blank reference)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    edges = cv2.Canny(gray, 50, 150)
    h_img, w_img = gray.shape[:2]
    ys, xs = np.where(edges > 0)
    if len(xs) == 0:
        cx, cy = w_img // 2, h_img // 2
    else:
        idx = int(rng.integers(0, len(xs)))
        cx, cy = int(xs[idx]), int(ys[idx])
    x = max(0, min(w_img - roi_size, cx - roi_size // 2))
    y = max(0, min(h_img - roi_size, cy - roi_size // 2))
    return (x, y, roi_size, roi_size)


def clamp_roi(
    roi_box: tuple[int, int, int, int], image_shape: tuple[int, int]
) -> tuple[int, int, int, int]:
    """Clip an (x, y, w, h) box to the image rectangle, dropping negative
    coordinates and trimming so x+w / y+h stay inside. Returns zero w or h
    if there is no intersection — callers should validate."""
    h_img, w_img = image_shape
    x, y, w, h = (int(roi_box[0]), int(roi_box[1]), int(roi_box[2]), int(roi_box[3]))
    if x < 0:
        w += x
        x = 0
    if y < 0:
        h += y
        y = 0
    w = max(0, min(w_img - x, w))
    h = max(0, min(h_img - y, h))
    return (x, y, w, h)


def validate_inputs(
    image: np.ndarray,
    direction: str,
    intensity: float,
    *,
    valid_directions: tuple[str, ...],
    direction_field_name: str = "flattening_direction",
) -> None:
    """Shared input validator. ``direction_field_name`` controls the error
    string so the message reads naturally for either op."""
    if not isinstance(image, np.ndarray):
        raise TypeError(f"image must be a numpy array, got {type(image).__name__}")
    if image.dtype != np.uint8:
        raise ValueError(f"image must be uint8, got {image.dtype}")
    if image.ndim not in (2, 3):
        raise ValueError(f"image must be 2D (gray) or 3D (BGR), got ndim={image.ndim}")
    if direction not in valid_directions:
        raise ValueError(
            f"{direction_field_name} must be one of {valid_directions}, "
            f"got {direction!r}"
        )
    if not (0.0 <= intensity <= 1.0):
        raise ValueError(f"intensity must be in [0.0, 1.0], got {intensity}")


def resolve_angle(direction: str, angle_degrees: float | None) -> float:
    """``angle_degrees`` wins when given; otherwise the legacy ``direction``
    field maps to 0° (horizontal) or 90° (vertical). Result is wrapped
    into the [0, 180) half-circle since both squeeze and stretch are
    axis-symmetric."""
    if angle_degrees is None:
        angle_degrees = 0.0 if direction == "horizontal" else 90.0
    angle = float(angle_degrees) % 180.0
    if angle < 0:
        angle += 180.0
    return angle
