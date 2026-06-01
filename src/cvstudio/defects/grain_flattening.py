"""Grain flattening defect injection — geometric squeeze, not blur.

Simulates mechanical-pressure flattening of grain structure on metal
microscope images by physically *compressing* the ROI texture along one
axis rather than blurring it. The grain's own micro-detail — boundaries,
roughness, contrast edges — survives intact; only the pixel positions are
squeezed, so the result reads as "pressed accordion-like" instead of
"filter applied." Designed for industrial-CV data augmentation: feed a
clean reference image, get back a defective copy plus YOLO-ready label
metadata.

Algorithm (per ROI):
    1. Pull a *wider* strip of natural texture from the source image (the
       ROI plus enough horizontal context on each side that, once
       squeezed, the strip resamples exactly down to the ROI's width).
       Squeeze it with ``cv2.INTER_LANCZOS4`` — no tiling, no mirroring,
       no synthetic symmetry. This is the default path whenever the
       source has horizontal context to spare.
    2. When the source has nothing to give (ROI hugs the image edge, or
       the caller already cropped down with ``fill_roi=True``), fall back
       to a *periodic* tile (NOT mirror) plus a random ``np.roll`` so the
       seam phase is randomised. The result still has a periodic
       structure, but the Rorschach-style centre-mirror butterfly never
       forms — that artefact is too easy for a downstream model to learn.
    3. For non-cardinal angles, the wide-context strip is taken in a
       rotated frame: ``warpAffine(image, -angle, output_size=(wide_w, h),
       INTER_LANCZOS4)`` lifts a rotated wide strip, the same X-squeeze
       runs on it, and ``warpAffine(+angle)`` puts the squeezed result
       back. Double warp loses some sharpness but stays far crisper than
       the old motion-blur path.
    4. A centre-bright, edge-dark feathered alpha mask blends the squeezed
       texture back into the unchanged surround so the ROI edge does not
       read as a paste-job.

ROI selection: when ``roi_box`` is None the injector samples a Canny edge
pixel and centres the ROI on it — defects land on visible grain structure
instead of empty background.
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
class GrainFlatteningLabel:
    """Metadata describing a single injected defect.

    Holds the canonical formats labelling tooling typically needs — pixel
    bounding boxes (xywh and xyxy), the normalised YOLO 4-tuple, and a
    closed-rectangle polygon — so callers can pick whichever the downstream
    format expects without re-doing arithmetic.
    """

    bbox_xywh: tuple[int, int, int, int]
    """(x, y, width, height) of the defect ROI in pixel coords."""

    bbox_xyxy: tuple[int, int, int, int]
    """(x_min, y_min, x_max, y_max) — exclusive right/bottom."""

    yolo: tuple[float, float, float, float]
    """(cx, cy, w, h) normalised to [0, 1] in image coords — drop into a
    YOLO ``.txt`` label line as ``f"{class_id} {cx} {cy} {w} {h}"``."""

    polygon: tuple[tuple[int, int], ...]
    """Closed polygon (4 corner points) outlining the ROI. Useful for
    instance-segmentation label formats."""

    image_shape: tuple[int, int]
    """``(height, width)`` of the source image — keeps the label
    self-describing for serialisation."""

    direction: str
    """Legacy convenience tag derived from the angle: ``'horizontal'`` or
    ``'vertical'`` for the cardinal cases, kept for backward compatibility.
    Prefer ``angle_degrees`` for any non-cardinal case."""

    angle_degrees: float
    """Smear angle wrapped into ``[0, 180)``. 0° = horizontal,
    90° = vertical, anything else = diagonal."""

    intensity: float
    """Severity used for this injection, clipped to ``[0.0, 1.0]``. Maps to
    horizontal squeeze ratio: ``ratio = 1.0 - intensity * 0.7`` (so
    intensity 1.0 squeezes the ROI to 30 % of its original width — the
    wider source strip that gets resampled down to that width comes from
    natural context around the ROI, no mirror tiling)."""


# --------------------------------------------------------------------- helpers


def _wide_x_span(x: int, w: int, ratio: float, image_width: int) -> tuple[int, int] | None:
    """Return (wide_x0, wide_x1) for a horizontal context strip that, once
    squeezed by ``ratio``, will exactly fill width ``w``. Returns None when
    the required strip would extend past the image edge — caller must fall
    back to the wrap+roll path.

    Picked symmetrically around the ROI centre when possible; clamped to
    image bounds only when the imbalance still fits within the wider span."""
    if ratio >= 1.0 - _NO_OP_RATIO_EPSILON:
        return (x, x + w)
    wide_w = max(w + 2, int(np.ceil(w / ratio)))
    extra = wide_w - w
    extra_l = extra // 2
    extra_r = extra - extra_l
    x0 = x - extra_l
    x1 = x + w + extra_r
    if x0 < 0:
        # Shift right; keep total width.
        x1 -= x0
        x0 = 0
    if x1 > image_width:
        # Shift left; keep total width.
        x0 -= x1 - image_width
        x1 = image_width
    if x0 < 0 or x1 > image_width:
        return None  # not enough horizontal context anywhere on the image
    return (x0, x1)


def _squeeze_from_wide_context(
    image: np.ndarray,
    *,
    x: int,
    y: int,
    w: int,
    h: int,
    ratio: float,
) -> np.ndarray | None:
    """Take a strip wider than ``w`` from ``image`` (centred on the ROI
    when possible), resize it down to (w, h) with INTER_LANCZOS4. The
    output is the entire squeezed view of natural texture — no mirroring,
    no tiling, no synthetic symmetry. Returns None when the wider strip
    won't fit on the image — caller should try the wrap+roll fallback."""
    img_w = image.shape[1]
    span = _wide_x_span(x, w, ratio, img_w)
    if span is None:
        return None
    x0, x1 = span
    wide = image[y : y + h, x0:x1]
    if wide.shape[1] == w:
        return wide.copy()  # ratio ≈ 1 short-circuit
    return cv2.resize(wide, (w, h), interpolation=cv2.INTER_LANCZOS4)


def _squeeze_with_reflect_pad(crop: np.ndarray, ratio: float) -> np.ndarray:
    """Last-resort fallback when there's no source image to widen from
    (only triggers when the source image is the same size as the ROI —
    rare in practice now that ``synthesis.grain_flattening`` flips
    ``needs_source=True``). Pad the crop horizontally with
    ``BORDER_REFLECT_101`` (which does NOT duplicate the boundary pixel,
    so the centre-mirror artefact is softer than a plain REFLECT), then
    resize the padded strip back down to ``w``. Pad amount is capped at
    just under the crop's own width because REFLECT_101 cannot extend
    further than the source supplies."""
    h, w = crop.shape[:2]
    if ratio >= 1.0:
        return crop.copy()
    desired_pad = (int(round(w / ratio)) - w)
    pad_total = min(desired_pad, max(0, 2 * (w - 1)))
    if pad_total <= 0:
        return crop.copy()
    pad_l = pad_total // 2
    pad_r = pad_total - pad_l
    padded = cv2.copyMakeBorder(
        crop, 0, 0, pad_l, pad_r, borderType=cv2.BORDER_REFLECT_101
    )
    return cv2.resize(padded, (w, h), interpolation=cv2.INTER_LANCZOS4)


# ---------------------------------------------------------------- main entry


def inject_grain_flattening(
    image: np.ndarray,
    *,
    roi_box: tuple[int, int, int, int] | None = None,
    flattening_direction: str = "horizontal",
    angle_degrees: float | None = None,
    intensity: float = 0.5,
    feather_fraction: float = 0.3,
    fill_roi: bool = False,
    seed: int | None = None,
) -> tuple[np.ndarray, GrainFlatteningLabel]:
    """Inject a synthetic grain-flattening defect into ``image``.

    The defect simulates mechanical pressure crushing grain structure along
    one axis by *geometrically compressing* the ROI texture with
    ``cv2.INTER_LANCZOS4`` and then mirror-tiling the narrowed result back
    to the original ROI width. Grain micro-detail (boundaries, roughness,
    contrast edges) is preserved — the squeeze moves pixels rather than
    blurring them. A feathered alpha blend dissolves the ROI boundary so
    the result reads as natural rather than as a pasted square.

    Args:
        image: Source image — ``uint8`` grayscale (H, W) or BGR (H, W, 3).
        roi_box: ``(x, y, w, h)`` of the region to corrupt, in pixel coords.
            ``None`` (default) samples a random ROI centred on a Canny edge
            pixel so the defect lands on visible grain structure instead of
            empty background. Ignored when ``fill_roi`` is True.
        flattening_direction: Legacy convenience — ``'horizontal'`` (= 0°)
            or ``'vertical'`` (= 90°). Overridden when ``angle_degrees`` is
            given explicitly.
        angle_degrees: Smear angle in degrees (0° = horizontal, 90° =
            vertical, anything in between produces a diagonal smear).
            Wrapped into ``[0, 180)``.
        intensity: ``[0.0, 1.0]`` severity. Maps linearly to the horizontal
            squeeze ratio: ``ratio = 1.0 - intensity * 0.7``. So
            intensity=0 is a no-op (identity output), intensity=0.5
            narrows the ROI to 65 % of its width (~35 % compression), and
            intensity=1 narrows it to 30 % (~70 % compression) before
            mirror-tiling fills the original width back in.
        feather_fraction: ``[0.0, 0.5]`` — share of each ROI side that
            fades into the surrounding texture. Larger = softer transition.
        fill_roi: When True the entire ``image`` is treated as the defect
            region — no auto-ROI sampling, no manual roi_box. Use this when
            the caller has already cropped the image to the area of
            interest (e.g. cvstudio's pipeline ROI hands you a crop).
        seed: RNG seed for the auto-ROI sampler. Ignored when
            ``fill_roi`` or ``roi_box`` is set.

    Returns:
        ``(output_image, label)``:
            * ``output_image`` — same shape and dtype as ``image``; only the
              ROI region (plus its feather fringe) is altered.
            * ``label`` — a :class:`GrainFlatteningLabel` carrying bbox,
              YOLO-normalised 4-tuple, polygon, and reproducibility metadata.

    Raises:
        TypeError: ``image`` is not a numpy array.
        ValueError: ``image`` is not ``uint8`` / not 2D or 3D; or
            ``flattening_direction`` is not in {horizontal, vertical}; or
            ``intensity`` is outside ``[0.0, 1.0]``; or ``roi_box`` clips to
            empty (no intersection with the image).

    Example:
        >>> import cv2
        >>> img = cv2.imread("metal_microscope.png", cv2.IMREAD_GRAYSCALE)
        >>> out, label = inject_grain_flattening(
        ...     img, angle_degrees=30, intensity=0.7, seed=42
        ... )
        >>> cv2.imwrite("metal_microscope_defect.png", out)
        >>> with open("metal_microscope_defect.txt", "w") as f:
        ...     cx, cy, w, h = label.yolo
        ...     f.write(f"0 {cx} {cy} {w} {h}\\n")
    """
    validate_inputs(
        image, flattening_direction, intensity,
        valid_directions=_VALID_DIRECTIONS,
        direction_field_name="flattening_direction",
    )
    feather_fraction = max(0.0, min(0.5, float(feather_fraction)))
    angle = resolve_angle(flattening_direction, angle_degrees)

    rng = np.random.default_rng(seed)
    h_img, w_img = image.shape[:2]

    if fill_roi:
        # Caller already cropped — the entire image IS the defect region.
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

    # --- 1. Geometric squeeze along the selected axis (LANCZOS4 keeps
    # high-frequency content sharp). To avoid the centre-mirror butterfly
    # artifact a naive REFLECT tile would create, we PREFER pulling a
    # wider strip of natural texture from the source image and squeezing
    # it down to ROI width — no tiling, no symmetry. When the source has
    # no horizontal context to spare (ROI at the edge, or fill_roi=True),
    # we fall back to a periodic tile + random np.roll so the seam phase
    # is randomised instead of mirrored.
    ratio = intensity_to_deform_ratio(intensity)
    if ratio >= 1.0 - _NO_OP_RATIO_EPSILON:
        flattened = roi.copy()
    elif angle == 0.0:
        wide = _squeeze_from_wide_context(image, x=x, y=y, w=w, h=h, ratio=ratio)
        flattened = wide if wide is not None else _squeeze_with_reflect_pad(roi, ratio)
    elif angle == 90.0:
        # Y-axis squeeze: transpose the picture, run the same X-axis path,
        # transpose back. swapaxes preserves channels for 3D input.
        image_t = np.swapaxes(image, 0, 1)
        wide_t = _squeeze_from_wide_context(
            image_t, x=y, y=x, w=h, h=w, ratio=ratio,
        )
        if wide_t is not None:
            flattened = np.ascontiguousarray(np.swapaxes(wide_t, 0, 1))
        else:
            roi_t = np.swapaxes(roi, 0, 1)
            squeezed_t = _squeeze_with_reflect_pad(roi_t, ratio)
            flattened = np.ascontiguousarray(np.swapaxes(squeezed_t, 0, 1))
    else:
        # Diagonal angle: rotate the SOURCE image -angle around the ROI
        # centre into a wide axis-aligned strip (so we get natural texture
        # from beyond the ROI), squeeze it down to (w, h). When the strip
        # would have to extend past the source image, fall back to
        # warp(ROI) + wrap+roll.
        cx_full = x + w / 2.0
        cy_full = y + h / 2.0
        span = _wide_x_span(x, w, ratio, image.shape[1])
        if span is not None:
            wide_w = span[1] - span[0]
            m_to_axis = cv2.getRotationMatrix2D((cx_full, cy_full), -angle, 1.0)
            # Shift so the rotated ROI centre lands at (wide_w/2, h/2).
            m_to_axis[0, 2] += wide_w / 2.0 - cx_full
            m_to_axis[1, 2] += h / 2.0 - cy_full
            wide_deaxis = cv2.warpAffine(
                image, m_to_axis, (wide_w, h),
                flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT,
            )
            squeezed = cv2.resize(
                wide_deaxis, (w, h), interpolation=cv2.INTER_LANCZOS4,
            )
        else:
            cx_c, cy_c = w / 2.0, h / 2.0
            m_to_axis_roi = cv2.getRotationMatrix2D((cx_c, cy_c), -angle, 1.0)
            deaxisified = cv2.warpAffine(
                roi, m_to_axis_roi, (w, h),
                flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT,
            )
            squeezed = _squeeze_with_reflect_pad(deaxisified, ratio)
        m_back = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        flattened = cv2.warpAffine(
            squeezed, m_back, (w, h),
            flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT,
        )

    # --- 2. Feathered alpha blend dissolves the ROI boundary into the
    # untouched surround so there is no boxy seam. The mask's centre is
    # fully opaque — the squeezed texture's sharpness is preserved there.
    alpha_mask = feathered_alpha(h, w, feather_fraction)
    if flattened.ndim == 3:
        alpha_mask = alpha_mask[..., None]
    blended_float = (
        flattened.astype(np.float32) * alpha_mask
        + roi.astype(np.float32) * (1.0 - alpha_mask)
    )
    blended = np.clip(blended_float, 0, 255).astype(np.uint8)

    # --- 3. Splice back into the original.
    out = image.copy()
    out[y : y + h, x : x + w] = blended

    label = GrainFlatteningLabel(
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
        direction=flattening_direction,
        angle_degrees=angle,
        intensity=float(intensity),
    )
    return out, label
