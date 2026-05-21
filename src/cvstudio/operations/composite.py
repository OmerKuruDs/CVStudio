"""Composite operations — multi-input ops that combine two images.

These are the first ops that exercise the DAG model. Each declares two input
ports. The chain auto-wires the first port to the previous pipeline node; the
second port stays unconnected until the user drags a wire to it in the UI.
Until then it falls back to the source image, so the op stays functional with
a sensible default.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from cvstudio.core.operation import OperationSpec, Parameter
from cvstudio.core.pipeline import coerce_to_match


def _blend(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    b_matched = coerce_to_match(b, a)
    if a.shape != b_matched.shape:
        b_matched = cv2.resize(b_matched, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR)
    return cv2.addWeighted(a, 1.0 - float(alpha), b_matched, float(alpha), 0.0)


def _blend_code(
    params: dict[str, Any], input_vars: tuple[str, ...], output_var: str
) -> list[str]:
    a, b = input_vars
    alpha = float(params["alpha"])
    return [
        f"_b = _coerce_to_match({b}, {a})",
        f"if _b.shape != {a}.shape:",
        f"    _b = cv2.resize(_b, ({a}.shape[1], {a}.shape[0]), interpolation=cv2.INTER_LINEAR)",
        f"{output_var} = cv2.addWeighted({a}, {1.0 - alpha}, _b, {alpha}, 0.0)",
    ]


def _apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if mask.shape != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    binary = (mask > 0).astype(np.uint8) * 255
    return cv2.bitwise_and(image, image, mask=binary)


def _apply_mask_code(
    _params: dict[str, Any], input_vars: tuple[str, ...], output_var: str
) -> list[str]:
    image, mask = input_vars
    return [
        f"_mask_in = {mask}",
        "if _mask_in.ndim == 3:",
        "    _mask_in = cv2.cvtColor(_mask_in, cv2.COLOR_BGR2GRAY)",
        f"if _mask_in.shape != {image}.shape[:2]:",
        f"    _mask_in = cv2.resize(_mask_in, ({image}.shape[1], {image}.shape[0]), "
        f"interpolation=cv2.INTER_NEAREST)",
        "_binary = (_mask_in > 0).astype('uint8') * 255",
        f"{output_var} = cv2.bitwise_and({image}, {image}, mask=_binary)",
    ]


def _difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    b_matched = coerce_to_match(b, a)
    if a.shape != b_matched.shape:
        b_matched = cv2.resize(b_matched, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR)
    return cv2.absdiff(a, b_matched)


def _difference_code(
    _params: dict[str, Any], input_vars: tuple[str, ...], output_var: str
) -> list[str]:
    a, b = input_vars
    return [
        f"_b = _coerce_to_match({b}, {a})",
        f"if _b.shape != {a}.shape:",
        f"    _b = cv2.resize(_b, ({a}.shape[1], {a}.shape[0]), interpolation=cv2.INTER_LINEAR)",
        f"{output_var} = cv2.absdiff({a}, _b)",
    ]


_INPAINT_METHODS: dict[str, int] = {
    "telea": cv2.INPAINT_TELEA,
    "ns": cv2.INPAINT_NS,
}


def _binarize_mask_for_inpaint(mask: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    """cv2.inpaint demands a uint8 single-channel mask matching the image's
    H,W. We accept whatever upstream wired in (color thresholding output, HSV
    range mask, freshly drawn mask, ...) and normalise it here."""
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if mask.shape != target_shape[:2]:
        mask = cv2.resize(mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8) * 255


def _inpaint(image: np.ndarray, mask: np.ndarray, method: str, radius: int) -> np.ndarray:
    flag = _INPAINT_METHODS.get(method, cv2.INPAINT_TELEA)
    binary_mask = _binarize_mask_for_inpaint(mask, image.shape)
    # cv2.inpaint requires uint8 3-channel or single-channel input. Float / 4-channel
    # inputs (rare in this pipeline but possible) get coerced to BGR uint8 first.
    work = image
    if work.dtype != np.uint8:
        work = np.clip(work, 0, 255).astype(np.uint8)
    if work.ndim == 3 and work.shape[2] == 4:
        work = cv2.cvtColor(work, cv2.COLOR_BGRA2BGR)
    return cv2.inpaint(work, binary_mask, float(radius), flag)


def _inpaint_code(
    params: dict[str, Any], input_vars: tuple[str, ...], output_var: str
) -> list[str]:
    image, mask = input_vars
    method = str(params["method"])
    radius = int(params["radius"])
    flag_name = "cv2.INPAINT_NS" if method == "ns" else "cv2.INPAINT_TELEA"
    return [
        f"_mask = {mask}",
        "if _mask.ndim == 3:",
        "    _mask = cv2.cvtColor(_mask, cv2.COLOR_BGR2GRAY)",
        f"if _mask.shape != {image}.shape[:2]:",
        f"    _mask = cv2.resize(_mask, ({image}.shape[1], {image}.shape[0]), "
        f"interpolation=cv2.INTER_NEAREST)",
        "_mask = (_mask > 0).astype('uint8') * 255",
        f"_img = {image}",
        "if _img.dtype != np.uint8:",
        "    _img = np.clip(_img, 0, 255).astype('uint8')",
        "if _img.ndim == 3 and _img.shape[2] == 4:",
        "    _img = cv2.cvtColor(_img, cv2.COLOR_BGRA2BGR)",
        f"{output_var} = cv2.inpaint(_img, _mask, {float(radius)}, {flag_name})",
    ]


def _mask_paste(
    dst: np.ndarray, mask: np.ndarray, binarize_threshold: int, fill_value: int
) -> np.ndarray:
    """Paint a constant fill_value into `dst` wherever `mask` exceeds the
    threshold. The synthesis use case: take a binarised defect-geometry mask
    and stamp it as a solid silhouette onto a clean substrate — the surrounding
    texture stays untouched, so the result reads as a real geometric defect
    rather than a seam-blended composite."""
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if mask.shape != dst.shape[:2]:
        mask = cv2.resize(mask, (dst.shape[1], dst.shape[0]), interpolation=cv2.INTER_NEAREST)
    binary = (mask > int(binarize_threshold)).astype(np.uint8) * 255
    result = dst.copy()
    # Scalar assignment broadcasts across channels so painted pixels stay
    # achromatic for both grayscale and multi-channel dst.
    result[binary > 0] = int(fill_value)
    return result


def _mask_paste_code(
    params: dict[str, Any], input_vars: tuple[str, ...], output_var: str
) -> list[str]:
    dst, mask = input_vars
    threshold = int(params["binarize_threshold"])
    fill = int(params["fill_value"])
    return [
        f"_mask = {mask}",
        "if _mask.ndim == 3:",
        "    _mask = cv2.cvtColor(_mask, cv2.COLOR_BGR2GRAY)",
        f"if _mask.shape != {dst}.shape[:2]:",
        f"    _mask = cv2.resize(_mask, ({dst}.shape[1], {dst}.shape[0]), "
        f"interpolation=cv2.INTER_NEAREST)",
        f"_binary = (_mask > {threshold}).astype('uint8') * 255",
        f"{output_var} = {dst}.copy()",
        f"{output_var}[_binary > 0] = {fill}",
    ]


_SEAMLESS_MODES: dict[str, int] = {
    "normal": cv2.NORMAL_CLONE,
    "mixed": cv2.MIXED_CLONE,
    "monochrome_transfer": cv2.MONOCHROME_TRANSFER,
}


def _to_bgr_uint8(image: np.ndarray) -> np.ndarray:
    """cv2.seamlessClone wants uint8 BGR for both src and dst."""
    work = image
    if work.dtype != np.uint8:
        work = np.clip(work, 0, 255).astype(np.uint8)
    if work.ndim == 2:
        work = cv2.cvtColor(work, cv2.COLOR_GRAY2BGR)
    elif work.shape[2] == 4:
        work = cv2.cvtColor(work, cv2.COLOR_BGRA2BGR)
    return work


def _mask_centroid(mask: np.ndarray) -> tuple[int, int] | None:
    """Centroid of the non-zero region of a single-channel uint8 mask, or
    None if the mask is empty."""
    moments = cv2.moments(mask)
    if moments["m00"] == 0:
        return None
    return (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))


def _clamp_center_to_fit(
    center: tuple[int, int],
    src_shape: tuple[int, ...],
    dst_shape: tuple[int, ...],
) -> tuple[int, int]:
    """seamlessClone fails if the bounding box of src centred on `center`
    extends past dst. Clamp the centre so the box always lies inside."""
    sh, sw = src_shape[:2]
    dh, dw = dst_shape[:2]
    half_w = sw // 2
    half_h = sh // 2
    # If src is larger than dst in either dimension there is no valid centre —
    # fall through to the natural cv2 error so the user sees the cause.
    cx = max(half_w + 1, min(dw - half_w - 1, int(center[0])))
    cy = max(half_h + 1, min(dh - half_h - 1, int(center[1])))
    return cx, cy


def _seamless_clone(
    src: np.ndarray,
    dst: np.ndarray,
    mask: np.ndarray,
    mode: str,
    auto_center: bool,
    center_x: int,
    center_y: int,
) -> np.ndarray:
    flag = _SEAMLESS_MODES.get(mode, cv2.NORMAL_CLONE)
    src_u8 = _to_bgr_uint8(src)
    dst_u8 = _to_bgr_uint8(dst)
    # cv2.seamlessClone requires the mask to share src's H,W and be single channel.
    mask_u8 = _binarize_mask_for_inpaint(mask, src_u8.shape)

    if auto_center:
        centroid = _mask_centroid(mask_u8)
        if centroid is None:
            # Empty mask → nothing to clone, return dst unchanged so the
            # pipeline keeps producing a valid image.
            return dst_u8
        center = centroid
    else:
        center = (int(center_x), int(center_y))

    center = _clamp_center_to_fit(center, src_u8.shape, dst_u8.shape)
    return cv2.seamlessClone(src_u8, dst_u8, mask_u8, center, flag)


def _seamless_clone_code(
    params: dict[str, Any], input_vars: tuple[str, ...], output_var: str
) -> list[str]:
    src, dst, mask = input_vars
    mode = str(params["mode"])
    auto = bool(params["auto_center"])
    cx = int(params["center_x"])
    cy = int(params["center_y"])
    flag_const = {
        "mixed": "cv2.MIXED_CLONE",
        "monochrome_transfer": "cv2.MONOCHROME_TRANSFER",
    }.get(mode, "cv2.NORMAL_CLONE")
    # Pre-compute the centre-selection block as either an auto branch (with
    # an empty-mask early return) or a plain literal assignment, so the
    # generated code stays a flat sequence.
    if auto:
        centre_lines = [
            "_m = cv2.moments(_mask)",
            "if _m['m00'] == 0:",
            f"    {output_var} = _dst",
            "else:",
            "    _cx = int(_m['m10'] / _m['m00'])",
            "    _cy = int(_m['m01'] / _m['m00'])",
            "    _sh, _sw = _src.shape[:2]",
            "    _dh, _dw = _dst.shape[:2]",
            "    _cx = max(_sw // 2 + 1, min(_dw - _sw // 2 - 1, _cx))",
            "    _cy = max(_sh // 2 + 1, min(_dh - _sh // 2 - 1, _cy))",
            f"    {output_var} = cv2.seamlessClone(_src, _dst, _mask, (_cx, _cy), {flag_const})",
        ]
    else:
        centre_lines = [
            f"_cx, _cy = {cx}, {cy}",
            "_sh, _sw = _src.shape[:2]",
            "_dh, _dw = _dst.shape[:2]",
            "_cx = max(_sw // 2 + 1, min(_dw - _sw // 2 - 1, _cx))",
            "_cy = max(_sh // 2 + 1, min(_dh - _sh // 2 - 1, _cy))",
            f"{output_var} = cv2.seamlessClone(_src, _dst, _mask, (_cx, _cy), {flag_const})",
        ]
    return [
        # --- normalise src
        f"_src = {src}",
        "if _src.dtype != np.uint8:",
        "    _src = np.clip(_src, 0, 255).astype('uint8')",
        "if _src.ndim == 2:",
        "    _src = cv2.cvtColor(_src, cv2.COLOR_GRAY2BGR)",
        "elif _src.shape[2] == 4:",
        "    _src = cv2.cvtColor(_src, cv2.COLOR_BGRA2BGR)",
        # --- normalise dst
        f"_dst = {dst}",
        "if _dst.dtype != np.uint8:",
        "    _dst = np.clip(_dst, 0, 255).astype('uint8')",
        "if _dst.ndim == 2:",
        "    _dst = cv2.cvtColor(_dst, cv2.COLOR_GRAY2BGR)",
        "elif _dst.shape[2] == 4:",
        "    _dst = cv2.cvtColor(_dst, cv2.COLOR_BGRA2BGR)",
        # --- mask must match src shape and be single-channel uint8
        f"_mask = {mask}",
        "if _mask.ndim == 3:",
        "    _mask = cv2.cvtColor(_mask, cv2.COLOR_BGR2GRAY)",
        "if _mask.shape != _src.shape[:2]:",
        "    _mask = cv2.resize(_mask, (_src.shape[1], _src.shape[0]), interpolation=cv2.INTER_NEAREST)",
        "_mask = (_mask > 0).astype('uint8') * 255",
        *centre_lines,
    ]


BLEND = OperationSpec(
    id="composite.blend",
    name="Blend",
    category="Composite",
    description=(
        "Alpha-blend two images. Input `a` is the chain-connected pipeline so far; "
        "wire `b` from any earlier node's output via drag-to-connect."
    ),
    parameters=(
        Parameter(
            name="alpha",
            kind="float",
            default=0.5,
            min=0.0,
            max=1.0,
            step=0.01,
            label="Alpha",
            description="Weight of input b. 0 = only a, 1 = only b.",
        ),
    ),
    func=_blend,
    code_export=_blend_code,
    input_ports=("a", "b"),
)


APPLY_MASK = OperationSpec(
    id="composite.apply_mask",
    name="Apply Mask",
    category="Composite",
    description=(
        "Keep pixels of input `image` where input `mask` is non-zero; zero "
        "elsewhere. Wire a thresholded image (or HSV-range mask) into `mask`."
    ),
    parameters=(),
    func=_apply_mask,
    code_export=_apply_mask_code,
    input_ports=("image", "mask"),
)


DIFFERENCE = OperationSpec(
    id="composite.difference",
    name="Difference",
    category="Composite",
    description=(
        "Absolute per-pixel difference between inputs `a` and `b`. Useful for "
        "highlighting what a transformation changed compared to the original."
    ),
    parameters=(),
    func=_difference,
    code_export=_difference_code,
    input_ports=("a", "b"),
)


INPAINT = OperationSpec(
    id="composite.inpaint",
    name="Inpaint",
    category="Composite",
    description=(
        "Reconstruct pixels of input `image` underneath input `mask` using "
        "neighbouring intact pixels. Wire any binary mask (defect region, "
        "scratch, occluder) into `mask`. Telea = fast marching, NS = "
        "Navier-Stokes (slower, often smoother on textured regions)."
    ),
    parameters=(
        Parameter(
            name="method",
            kind="choice",
            default="telea",
            choices=("telea", "ns"),
            label="Method",
            description="Telea (fast marching) or Navier-Stokes inpainting.",
        ),
        Parameter(
            name="radius",
            kind="int",
            default=3,
            min=1,
            max=30,
            step=1,
            label="Radius",
            description="Pixel neighbourhood radius considered for each inpainted pixel.",
        ),
    ),
    func=_inpaint,
    code_export=_inpaint_code,
    input_ports=("image", "mask"),
)


MASK_PASTE = OperationSpec(
    id="composite.mask_paste",
    name="Mask Paste",
    category="Composite",
    description=(
        "Paint a constant `fill_value` into input `dst` wherever input `mask` "
        "is above the threshold. Unlike Blend / Seamless Clone this is a hard "
        "stencil — the substrate's texture stays untouched, so it reads as a "
        "real geometric mark (defect, hole, scratch) rather than a transferred "
        "patch."
    ),
    parameters=(
        Parameter(
            name="binarize_threshold",
            kind="int",
            default=127,
            min=0,
            max=255,
            step=1,
            label="Mask Threshold",
            description="Mask pixels strictly above this value are painted.",
        ),
        Parameter(
            name="fill_value",
            kind="int",
            default=0,
            min=0,
            max=255,
            step=1,
            label="Fill Value",
            description="Intensity used for painted pixels (broadcast across channels).",
        ),
    ),
    func=_mask_paste,
    code_export=_mask_paste_code,
    input_ports=("dst", "mask"),
)


SEAMLESS_CLONE = OperationSpec(
    id="composite.seamless_clone",
    name="Seamless Clone",
    category="Composite",
    description=(
        "Poisson image editing — paste the mask-shaped region of input `src` "
        "into input `dst` so its gradients blend into the destination's lighting "
        "and texture. Wire `src` (donor patch), `dst` (background), and `mask` "
        "(region of src to copy). `auto_center` puts the patch at the mask's "
        "centroid; turn it off to place the patch at (center_x, center_y) "
        "manually. The centre is clamped so the patch always fits inside dst."
    ),
    parameters=(
        Parameter(
            name="mode",
            kind="choice",
            default="normal",
            choices=("normal", "mixed", "monochrome_transfer"),
            label="Mode",
            description=(
                "Normal = preserve src colours. Mixed = pick the stronger "
                "gradient between src and dst (good for textured dst). "
                "Monochrome transfer = ignore src colour, copy only luminance."
            ),
        ),
        Parameter(
            name="auto_center",
            kind="bool",
            default=True,
            label="Auto centre",
            description="Place the patch at the mask's centroid.",
        ),
        Parameter(
            name="center_x",
            kind="int",
            default=0,
            min=0,
            max=8192,
            step=1,
            label="Centre X",
            description="Destination x coordinate (used only when Auto centre is off).",
        ),
        Parameter(
            name="center_y",
            kind="int",
            default=0,
            min=0,
            max=8192,
            step=1,
            label="Centre Y",
            description="Destination y coordinate (used only when Auto centre is off).",
        ),
    ),
    func=_seamless_clone,
    code_export=_seamless_clone_code,
    input_ports=("src", "dst", "mask"),
)


ALL: tuple[OperationSpec, ...] = (
    BLEND,
    APPLY_MASK,
    DIFFERENCE,
    INPAINT,
    MASK_PASTE,
    SEAMLESS_CLONE,
)
