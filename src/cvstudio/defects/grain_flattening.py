"""Grain flattening defect injection.

Simulates mechanical-pressure flattening of grain structure on metal
microscope images — directional smearing, grain merging, and a local
matte/contrast drop, all blended back through a feathered alpha mask so
the patched region never reads as a square paste-job. Designed for
industrial-CV data augmentation: feed a clean reference image, get back
a defective copy plus YOLO-ready label metadata.

Algorithm (per ROI):
    1. Directional motion blur smears grain texture along the flattening axis.
    2. Directional morphological close merges adjacent grain boundaries.
    3. ``convertScaleAbs`` drops local contrast (matte appearance).
    4. A centre-bright, edge-dark Gaussian alpha mask blends the manipulated
       texture back into the unchanged surround.

ROI selection: when ``roi_box`` is None the injector samples a Canny edge
pixel and centres the ROI on it — defects land on visible grain structure
instead of empty background.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

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
    """Severity used for this injection, clipped to ``[0.0, 1.0]``."""


# --------------------------------------------------------------------- helpers


def _motion_blur_kernel(size: int, angle_degrees: float) -> np.ndarray:
    """Single-line averaging kernel rotated to ``angle_degrees`` — the
    canonical 'motion blur' filter at an arbitrary angle. 0° = horizontal,
    90° = vertical, any value in between produces a diagonal smear.
    Already normalised so ``filter2D`` preserves global brightness."""
    kernel = np.zeros((size, size), dtype=np.float32)
    centre = (size - 1) / 2
    half = (size - 1) / 2
    angle_rad = np.deg2rad(angle_degrees)
    dx = np.cos(angle_rad)
    dy = np.sin(angle_rad)
    x0 = int(round(centre - half * dx))
    y0 = int(round(centre - half * dy))
    x1 = int(round(centre + half * dx))
    y1 = int(round(centre + half * dy))
    cv2.line(kernel, (x0, y0), (x1, y1), 1.0, thickness=1)
    if kernel.sum() == 0:
        kernel[size // 2, size // 2] = 1.0
    kernel /= kernel.sum()
    return kernel


def _directional_morph_kernel(size: int, angle_degrees: float) -> np.ndarray:
    """Thin rectangular structuring element rotated to ``angle_degrees``.
    cv2 has no native rotated SE, so we draw a horizontal line on a square
    canvas and warpAffine it — the morph close then only bridges grains
    along the chosen axis."""
    if size % 2 == 0:
        size += 1
    canvas = np.zeros((size, size), dtype=np.uint8)
    mid = size // 2
    cv2.line(canvas, (0, mid), (size - 1, mid), 1, thickness=1)
    # Negative angle so the line ends up at +angle_degrees from horizontal
    # under OpenCV's (counter-intuitive) image-coords rotation convention.
    rot = cv2.getRotationMatrix2D((mid, mid), -float(angle_degrees), 1.0)
    rotated = cv2.warpAffine(canvas, rot, (size, size), flags=cv2.INTER_LINEAR)
    se = (rotated > 0).astype(np.uint8)
    if not se.any():
        se[mid, mid] = 1
    return se


def _intensity_to_kernel_size(
    intensity: float,
    *,
    min_size: int = 3,
    max_size: int = 21,
    image_min_dim: int | None = None,
) -> int:
    """Map intensity ∈ [0, 1] to an odd kernel size in [min_size, max_size].
    Forced odd because OpenCV kernels are typically odd-sided. When
    ``image_min_dim`` is set the result is also capped at roughly a
    quarter of the shorter side so tiny ROIs do not get blanketed by an
    oversize kernel."""
    span = max_size - min_size
    raw = int(round(min_size + max(0.0, min(1.0, intensity)) * span))
    if image_min_dim is not None:
        soft_cap = max(min_size, image_min_dim // 4)
        raw = min(raw, soft_cap)
    return max(min_size, raw) | 1


def _feathered_alpha(height: int, width: int, fraction: float) -> np.ndarray:
    """Centre-bright, edge-dark alpha mask. The inner rectangle is fully
    opaque (1.0); a Gaussian blur softens the transition to the ROI edge so
    the blended patch fades into the surrounding texture instead of leaving
    a boxy seam. ``fraction`` ∈ [0, 0.5] sets how much of each side gets
    feathered (0.3 = inner 40% solid, outer 30% on each side fading)."""
    fraction = max(0.0, min(0.5, float(fraction)))
    margin_y = max(1, int(round(height * fraction)))
    margin_x = max(1, int(round(width * fraction)))
    inner = np.zeros((height, width), dtype=np.float32)
    inner[margin_y : height - margin_y, margin_x : width - margin_x] = 1.0
    # Blur radius scales with the feather margin so smaller ROIs get
    # proportionally tighter fades.
    blur_size = max(margin_x, margin_y) | 1
    blur_size = max(3, blur_size)
    return cv2.GaussianBlur(inner, (blur_size, blur_size), 0)


def _pick_random_roi_near_edge(
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


def _clamp_roi(
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


def _validate(image: np.ndarray, flattening_direction: str, intensity: float) -> None:
    if not isinstance(image, np.ndarray):
        raise TypeError(f"image must be a numpy array, got {type(image).__name__}")
    if image.dtype != np.uint8:
        raise ValueError(f"image must be uint8, got {image.dtype}")
    if image.ndim not in (2, 3):
        raise ValueError(f"image must be 2D (gray) or 3D (BGR), got ndim={image.ndim}")
    if flattening_direction not in _VALID_DIRECTIONS:
        raise ValueError(
            f"flattening_direction must be one of {_VALID_DIRECTIONS}, "
            f"got {flattening_direction!r}"
        )
    if not (0.0 <= intensity <= 1.0):
        raise ValueError(f"intensity must be in [0.0, 1.0], got {intensity}")


def _resolve_angle(flattening_direction: str, angle_degrees: float | None) -> float:
    """``angle_degrees`` wins when given; otherwise legacy ``flattening_direction``
    maps to 0° (horizontal) or 90° (vertical). Result is wrapped into the
    [0, 180) half-circle since the smear is direction-symmetric."""
    if angle_degrees is None:
        angle_degrees = 0.0 if flattening_direction == "horizontal" else 90.0
    angle = float(angle_degrees) % 180.0
    if angle < 0:
        angle += 180.0
    return angle


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
    one axis: directional motion blur smears the texture, a directional
    morph-close merges neighbouring grains, a small contrast drop mats the
    region, and a feathered alpha blend dissolves the ROI boundary so the
    result reads as natural rather than as a pasted square.

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
        intensity: ``[0.0, 1.0]`` severity. Drives the motion-blur kernel
            size (3 → 21 px), the morph-close kernel size, and the contrast
            drop (``alpha`` from 1.00 down to 0.85). Kernel sizes are also
            soft-capped at roughly a quarter of the ROI's shorter side so
            tiny ROIs do not get blanketed.
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
    _validate(image, flattening_direction, intensity)
    feather_fraction = max(0.0, min(0.5, float(feather_fraction)))
    angle = _resolve_angle(flattening_direction, angle_degrees)

    rng = np.random.default_rng(seed)
    h_img, w_img = image.shape[:2]

    if fill_roi:
        # Caller already cropped — the entire image IS the defect region.
        roi_box = (0, 0, w_img, h_img)
    elif roi_box is None:
        default_size = max(24, min(h_img, w_img) // 6)
        roi_box = _pick_random_roi_near_edge(image, default_size, rng)

    x, y, w, h = _clamp_roi(roi_box, (h_img, w_img))
    if w <= 0 or h <= 0:
        raise ValueError(
            f"roi_box {roi_box!r} has no overlap with image of shape "
            f"{(h_img, w_img)}"
        )

    roi = image[y : y + h, x : x + w].copy()

    # --- 1. Directional motion blur smears grain texture along the axis.
    # Kernel size is soft-capped relative to the ROI so small mouse-drawn
    # boxes do not get blanketed by an outsize kernel.
    motion_size = _intensity_to_kernel_size(
        intensity, min_size=3, max_size=21, image_min_dim=min(h, w),
    )
    motion_kernel = _motion_blur_kernel(motion_size, angle)
    smeared = cv2.filter2D(roi, ddepth=-1, kernel=motion_kernel)

    # --- 2. Directional morph close merges adjacent grain boundaries.
    morph_size = max(3, (motion_size // 2) | 1)
    morph_kernel = _directional_morph_kernel(morph_size, angle)
    flattened = cv2.morphologyEx(smeared, cv2.MORPH_CLOSE, morph_kernel)

    # --- 3. Local contrast drop produces the matte / dulled appearance that
    # accompanies mechanically-flattened grains (less directional reflection).
    alpha = 1.0 - 0.15 * intensity
    beta = 8.0 * intensity
    matted = cv2.convertScaleAbs(flattened, alpha=alpha, beta=beta)

    # --- 4. Feathered alpha blend dissolves the ROI boundary into the
    # untouched surround so there is no boxy seam.
    alpha_mask = _feathered_alpha(h, w, feather_fraction)
    if matted.ndim == 3:
        alpha_mask = alpha_mask[..., None]
    blended_float = (
        matted.astype(np.float32) * alpha_mask
        + roi.astype(np.float32) * (1.0 - alpha_mask)
    )
    blended = np.clip(blended_float, 0, 255).astype(np.uint8)

    # --- 5. Splice back into the original.
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
