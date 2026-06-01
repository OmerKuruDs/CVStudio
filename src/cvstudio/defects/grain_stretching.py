"""Grain stretching defect injection — geometric stretch, not blur.

The inverse of ``grain_flattening``: pulls a NARROWER strip of natural
texture from around the ROI and resamples it UP to the ROI's width with
``cv2.INTER_LANCZOS4``. Pixels end up physically wider along the selected
axis by ``1/ratio``. Grain micro-detail (boundaries, roughness, contrast
edges) survives intact because LANCZOS4 keeps high-frequency content
sharp under upsampling.

Algorithm (per ROI):
    1. Take a strip from the source centred on the ROI but ``ratio``×
       narrower along the squeeze axis. Because the strip is narrower
       than the ROI it always fits — no bounds-fallback ever needed.
    2. ``cv2.resize`` the strip up to the full ROI extent with LANCZOS4.
       This is the stretch.
    3. For diagonal angles, first warp the source -angle into an axis-
       aligned ROI buffer (with the same wide source as a context window
       so border pixels look natural), take the narrow centre slice,
       resize up to (w, h), and warp +angle back.
    4. Feathered alpha blend at the ROI boundary so the stretched patch
       fades into the unchanged surround.

API surface mirrors ``inject_grain_flattening``: same parameters, a
sibling ``GrainStretchingLabel`` dataclass, and the same UI op semantics
(``synthesis.grain_stretching`` with ``needs_source=True``).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from cvstudio.defects._common import (
    _NO_OP_RATIO_EPSILON,
    clamp_roi,
    feathered_alpha,
    intensity_to_deform_ratio,
    pick_random_roi_near_edge,
    resolve_angle,
    validate_inputs,
)

_VALID_DIRECTIONS = ("horizontal", "vertical")


@dataclass(frozen=True)
class GrainStretchingLabel:
    """Metadata describing a single injected stretch defect.

    Distinct class from ``GrainFlatteningLabel`` so downstream label
    routing can tell the two defect types apart, even though every field
    has the same shape and meaning.
    """

    bbox_xywh: tuple[int, int, int, int]
    """(x, y, width, height) of the defect ROI in pixel coords."""

    bbox_xyxy: tuple[int, int, int, int]
    """(x_min, y_min, x_max, y_max) — exclusive right/bottom."""

    yolo: tuple[float, float, float, float]
    """(cx, cy, w, h) normalised to [0, 1] in image coords — drop into a
    YOLO ``.txt`` label line as ``f"{class_id} {cx} {cy} {w} {h}"``."""

    polygon: tuple[tuple[int, int], ...]
    """Closed polygon (4 corner points) outlining the ROI."""

    image_shape: tuple[int, int]
    """``(height, width)`` of the source image."""

    direction: str
    """Legacy convenience tag derived from the angle: ``'horizontal'`` or
    ``'vertical'`` for the cardinal cases. Prefer ``angle_degrees``."""

    angle_degrees: float
    """Stretch axis in degrees, wrapped into ``[0, 180)``. 0° = horizontal,
    90° = vertical, anything else = diagonal."""

    intensity: float
    """Severity in ``[0.0, 1.0]``. Maps to stretch ratio: pixels widen by
    ``1 / (1 - intensity * 0.7)`` along the selected axis (so intensity
    1.0 makes them ~3.3× wider, intensity 0.5 → ~1.5×)."""


# --------------------------------------------------------------------- helpers


def _narrow_x_span(x: int, w: int, ratio: float, image_width: int) -> tuple[int, int]:
    """Return (x0, x1) for a horizontal source strip ``ratio×`` narrower
    than the ROI, centred on the ROI. Always fits inside the image —
    narrow strips trivially fit wherever the ROI does."""
    narrow_w = max(2, int(round(w * ratio)))
    if narrow_w >= w:
        return (x, x + w)  # ratio ≈ 1 corner case
    cx = x + w / 2.0
    x0 = int(round(cx - narrow_w / 2.0))
    # Clamp to image — the narrow strip is always smaller than the ROI,
    # so the clamp is purely defensive in pathological cases (the ROI is
    # already validated to live inside the image).
    x0 = max(0, min(image_width - narrow_w, x0))
    return (x0, x0 + narrow_w)


def _stretch_from_narrow_centre(
    image: np.ndarray, *, x: int, y: int, w: int, h: int, ratio: float
) -> np.ndarray:
    """Take a ``ratio×`` narrower strip from ``image`` centred on the ROI
    and resample it back up to ``(w, h)`` with LANCZOS4."""
    x0, x1 = _narrow_x_span(x, w, ratio, image.shape[1])
    crop = image[y : y + h, x0:x1]
    if crop.shape[1] == w:
        return crop.copy()
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LANCZOS4)


# ---------------------------------------------------------------- main entry


def inject_grain_stretching(
    image: np.ndarray,
    *,
    roi_box: tuple[int, int, int, int] | None = None,
    stretching_direction: str = "horizontal",
    angle_degrees: float | None = None,
    intensity: float = 0.5,
    feather_fraction: float = 0.3,
    fill_roi: bool = False,
    seed: int | None = None,
) -> tuple[np.ndarray, GrainStretchingLabel]:
    """Inject a synthetic grain-stretching defect into ``image``.

    The defect is the geometric inverse of ``grain_flattening``: a strip
    of natural texture narrower than the ROI is pulled from the source
    and resampled up to ROI width with LANCZOS4, so pixels widen along
    the selected axis. Grain micro-detail survives because LANCZOS4 does
    not soften high-frequency content under upsampling. A feathered alpha
    blend dissolves the ROI boundary so the result reads as natural.

    Args:
        image: Source image — ``uint8`` grayscale (H, W) or BGR (H, W, 3).
        roi_box: ``(x, y, w, h)`` of the region to corrupt, in pixel coords.
            ``None`` samples a random ROI centred on a Canny edge pixel.
            Ignored when ``fill_roi`` is True.
        stretching_direction: Legacy convenience — ``'horizontal'`` (= 0°)
            or ``'vertical'`` (= 90°). Overridden when ``angle_degrees`` is
            given explicitly.
        angle_degrees: Stretch axis in degrees (0° = horizontal,
            90° = vertical, anything in between = diagonal). Wrapped into
            ``[0, 180)``.
        intensity: ``[0.0, 1.0]`` severity. Maps linearly to the stretch
            ratio: ``ratio = 1.0 - intensity * 0.7``. intensity 0 = no-op,
            intensity 0.5 ≈ 1.5× widening, intensity 1.0 ≈ 3.3× widening.
        feather_fraction: ``[0.0, 0.5]`` — share of each ROI side that
            fades into the surrounding texture.
        fill_roi: When True the entire ``image`` is treated as the defect
            region — no auto-ROI sampling, no manual roi_box. Use when the
            caller has already cropped to the area of interest.
        seed: RNG seed for the auto-ROI sampler. Ignored when
            ``fill_roi`` or ``roi_box`` is set.

    Returns:
        ``(output_image, label)``:
            * ``output_image`` — same shape and dtype as ``image``.
            * ``label`` — a :class:`GrainStretchingLabel`.

    Raises:
        TypeError / ValueError on the same invariants ``inject_grain_flattening``
        enforces (uint8 input, intensity in [0, 1], etc.).
    """
    validate_inputs(
        image, stretching_direction, intensity,
        valid_directions=_VALID_DIRECTIONS,
        direction_field_name="stretching_direction",
    )
    feather_fraction = max(0.0, min(0.5, float(feather_fraction)))
    angle = resolve_angle(stretching_direction, angle_degrees)

    rng = np.random.default_rng(seed)
    h_img, w_img = image.shape[:2]

    if fill_roi:
        roi_box = (0, 0, w_img, h_img)
    elif roi_box is None:
        default_size = max(24, min(h_img, w_img) // 6)
        roi_box = pick_random_roi_near_edge(image, default_size, rng)

    x, y, w, h = clamp_roi(roi_box, (h_img, w_img))
    if w <= 0 or h <= 0:
        raise ValueError(
            f"roi_box {roi_box!r} has no overlap with image of shape "
            f"{(h_img, w_img)}"
        )

    roi = image[y : y + h, x : x + w].copy()

    ratio = intensity_to_deform_ratio(intensity)
    if ratio >= 1.0 - _NO_OP_RATIO_EPSILON:
        stretched = roi.copy()
    elif angle == 0.0:
        stretched = _stretch_from_narrow_centre(
            image, x=x, y=y, w=w, h=h, ratio=ratio,
        )
    elif angle == 90.0:
        # Y-axis stretch: transpose to put the squeeze axis on X, run the
        # same path, transpose back. swapaxes preserves channels for 3D.
        image_t = np.swapaxes(image, 0, 1)
        stretched_t = _stretch_from_narrow_centre(
            image_t, x=y, y=x, w=h, h=w, ratio=ratio,
        )
        stretched = np.ascontiguousarray(np.swapaxes(stretched_t, 0, 1))
    else:
        # Diagonal: rotate the SOURCE image -angle around the ROI centre
        # into an axis-aligned (w, h) ROI buffer (LANCZOS4 keeps edges
        # sharp; BORDER_REFLECT only kicks in if the ROI sits at the very
        # image edge), take the narrow centre slice, stretch up to (w, h),
        # and warp back.
        cx_full = x + w / 2.0
        cy_full = y + h / 2.0
        m_to_axis = cv2.getRotationMatrix2D((cx_full, cy_full), -angle, 1.0)
        m_to_axis[0, 2] += w / 2.0 - cx_full
        m_to_axis[1, 2] += h / 2.0 - cy_full
        deaxisified = cv2.warpAffine(
            image, m_to_axis, (w, h),
            flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT,
        )
        narrow_w = max(2, int(round(w * ratio)))
        start = (w - narrow_w) // 2
        narrow_slice = deaxisified[:, start : start + narrow_w]
        stretched_axis = cv2.resize(
            narrow_slice, (w, h), interpolation=cv2.INTER_LANCZOS4,
        )
        m_back = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        stretched = cv2.warpAffine(
            stretched_axis, m_back, (w, h),
            flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT,
        )

    alpha_mask = feathered_alpha(h, w, feather_fraction)
    if stretched.ndim == 3:
        alpha_mask = alpha_mask[..., None]
    blended_float = (
        stretched.astype(np.float32) * alpha_mask
        + roi.astype(np.float32) * (1.0 - alpha_mask)
    )
    blended = np.clip(blended_float, 0, 255).astype(np.uint8)

    out = image.copy()
    out[y : y + h, x : x + w] = blended

    label = GrainStretchingLabel(
        bbox_xywh=(x, y, w, h),
        bbox_xyxy=(x, y, x + w, y + h),
        yolo=(
            (x + w / 2.0) / w_img,
            (y + h / 2.0) / h_img,
            w / w_img,
            h / h_img,
        ),
        polygon=((x, y), (x + w, y), (x + w, y + h), (x, y + h)),
        image_shape=(h_img, w_img),
        direction=stretching_direction,
        angle_degrees=angle,
        intensity=float(intensity),
    )
    return out, label
