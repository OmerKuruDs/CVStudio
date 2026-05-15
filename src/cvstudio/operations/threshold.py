"""Threshold operations.

All thresholds need a single-channel input; 3-channel input is converted to
grayscale at the boundary. Output is single-channel uint8 — downstream ops that
need 3 channels should add a grayscale-to-BGR conversion after.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from cvsandbox.core.operation import OperationSpec, Parameter


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def _binary_threshold(image: np.ndarray, thresh: int, maxval: int, inverse: bool) -> np.ndarray:
    gray = _to_gray(image)
    mode = cv2.THRESH_BINARY_INV if inverse else cv2.THRESH_BINARY
    _, out = cv2.threshold(gray, float(thresh), float(maxval), mode)
    return out


def _otsu_threshold(image: np.ndarray, maxval: int, inverse: bool) -> np.ndarray:
    gray = _to_gray(image)
    mode = (cv2.THRESH_BINARY_INV if inverse else cv2.THRESH_BINARY) | cv2.THRESH_OTSU
    _, out = cv2.threshold(gray, 0, float(maxval), mode)
    return out


_ADAPTIVE_METHODS = {
    "Mean": cv2.ADAPTIVE_THRESH_MEAN_C,
    "Gaussian": cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
}


def _to_gray_line(in_var: str, out_var: str) -> str:
    return (
        f"{out_var} = cv2.cvtColor({in_var}, cv2.COLOR_BGR2GRAY) "
        f"if {in_var}.ndim == 3 else {in_var}"
    )


def _binary_code(
    params: dict[str, Any], input_vars: tuple[str, ...], output_var: str
) -> list[str]:
    (a,) = input_vars
    mode = "cv2.THRESH_BINARY_INV" if params["inverse"] else "cv2.THRESH_BINARY"
    return [
        _to_gray_line(a, output_var),
        f"_, {output_var} = cv2.threshold({output_var}, {float(params['thresh'])}, "
        f"{float(params['maxval'])}, {mode})",
    ]


def _otsu_code(
    params: dict[str, Any], input_vars: tuple[str, ...], output_var: str
) -> list[str]:
    (a,) = input_vars
    base = "cv2.THRESH_BINARY_INV" if params["inverse"] else "cv2.THRESH_BINARY"
    return [
        _to_gray_line(a, output_var),
        f"_, {output_var} = cv2.threshold({output_var}, 0, {float(params['maxval'])}, "
        f"{base} | cv2.THRESH_OTSU)",
    ]


_ADAPTIVE_CODE_CONSTS = {
    "Mean": "cv2.ADAPTIVE_THRESH_MEAN_C",
    "Gaussian": "cv2.ADAPTIVE_THRESH_GAUSSIAN_C",
}


def _adaptive_code(
    params: dict[str, Any], input_vars: tuple[str, ...], output_var: str
) -> list[str]:
    (a,) = input_vars
    method_const = _ADAPTIVE_CODE_CONSTS[params["method"]]
    block = max(3, int(params["block_size"]) | 1)
    mode = "cv2.THRESH_BINARY_INV" if params["inverse"] else "cv2.THRESH_BINARY"
    return [
        _to_gray_line(a, output_var),
        f"{output_var} = cv2.adaptiveThreshold({output_var}, {float(params['maxval'])}, "
        f"{method_const}, {mode}, {block}, {float(params['c'])})",
    ]


def _adaptive_threshold(
    image: np.ndarray,
    maxval: int,
    method: str,
    block_size: int,
    c: int,
    inverse: bool,
) -> np.ndarray:
    gray = _to_gray(image)
    block = max(3, int(block_size) | 1)  # must be odd, >= 3
    mode = cv2.THRESH_BINARY_INV if inverse else cv2.THRESH_BINARY
    return cv2.adaptiveThreshold(
        gray,
        float(maxval),
        _ADAPTIVE_METHODS[method],
        mode,
        block,
        float(c),
    )


BINARY_THRESHOLD = OperationSpec(
    id="threshold.binary",
    name="Binary Threshold",
    category="Threshold",
    description="Pixels above `thresh` become `maxval`, others become 0.",
    parameters=(
        Parameter(name="thresh", kind="int", default=127, min=0, max=255, label="Threshold"),
        Parameter(name="maxval", kind="int", default=255, min=0, max=255, label="Max value"),
        Parameter(name="inverse", kind="bool", default=False, label="Invert"),
    ),
    func=_binary_threshold,
    code_export=_binary_code,
)


OTSU_THRESHOLD = OperationSpec(
    id="threshold.otsu",
    name="Otsu Threshold",
    category="Threshold",
    description="Picks the threshold automatically from the image histogram (bimodal assumption).",
    parameters=(
        Parameter(name="maxval", kind="int", default=255, min=0, max=255, label="Max value"),
        Parameter(name="inverse", kind="bool", default=False, label="Invert"),
    ),
    func=_otsu_threshold,
    code_export=_otsu_code,
)


ADAPTIVE_THRESHOLD = OperationSpec(
    id="threshold.adaptive",
    name="Adaptive Threshold",
    category="Threshold",
    description="Threshold computed per-region. Handles uneven illumination.",
    parameters=(
        Parameter(name="maxval", kind="int", default=255, min=0, max=255, label="Max value"),
        Parameter(
            name="method",
            kind="choice",
            default="Gaussian",
            choices=tuple(_ADAPTIVE_METHODS.keys()),
            label="Method",
        ),
        Parameter(
            name="block_size",
            kind="kernel_size",
            default=11,
            min=3,
            max=99,
            step=2,
            label="Block size",
            description="Odd, ≥3. Neighborhood used to compute the local threshold.",
        ),
        Parameter(
            name="c",
            kind="int",
            default=2,
            min=-50,
            max=50,
            label="C",
            description="Constant subtracted from the local mean.",
        ),
        Parameter(name="inverse", kind="bool", default=False, label="Invert"),
    ),
    func=_adaptive_threshold,
    code_export=_adaptive_code,
)


def _triangle_threshold(image: np.ndarray, maxval: int, inverse: bool) -> np.ndarray:
    gray = _to_gray(image)
    mode = (cv2.THRESH_BINARY_INV if inverse else cv2.THRESH_BINARY) | cv2.THRESH_TRIANGLE
    _, out = cv2.threshold(gray, 0, float(maxval), mode)
    return out


def _triangle_code(
    params: dict[str, Any], input_vars: tuple[str, ...], output_var: str
) -> list[str]:
    (a,) = input_vars
    base = "cv2.THRESH_BINARY_INV" if params["inverse"] else "cv2.THRESH_BINARY"
    return [
        _to_gray_line(a, output_var),
        f"_, {output_var} = cv2.threshold({output_var}, 0, {float(params['maxval'])}, "
        f"{base} | cv2.THRESH_TRIANGLE)",
    ]


def _in_range_bgr(
    image: np.ndarray,
    b_low: int,
    b_high: int,
    g_low: int,
    g_high: int,
    r_low: int,
    r_high: int,
) -> np.ndarray:
    if image.ndim == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        bgr = image
    lower = np.array(
        [min(int(b_low), int(b_high)), min(int(g_low), int(g_high)), min(int(r_low), int(r_high))],
        dtype=np.uint8,
    )
    upper = np.array(
        [max(int(b_low), int(b_high)), max(int(g_low), int(g_high)), max(int(r_low), int(r_high))],
        dtype=np.uint8,
    )
    return cv2.inRange(bgr, lower, upper)


def _in_range_code(
    params: dict[str, Any], input_vars: tuple[str, ...], output_var: str
) -> list[str]:
    (a,) = input_vars
    bl, bh = int(params["b_low"]), int(params["b_high"])
    gl, gh = int(params["g_low"]), int(params["g_high"])
    rl, rh = int(params["r_low"]), int(params["r_high"])
    return [
        _to_bgr_line(a, "_bgr"),
        f"_lower = np.array(({min(bl, bh)}, {min(gl, gh)}, {min(rl, rh)}), dtype=np.uint8)",
        f"_upper = np.array(({max(bl, bh)}, {max(gl, gh)}, {max(rl, rh)}), dtype=np.uint8)",
        f"{output_var} = cv2.inRange(_bgr, _lower, _upper)",
    ]


def _to_bgr_line(in_var: str, out_var: str) -> str:
    return (
        f"{out_var} = cv2.cvtColor({in_var}, cv2.COLOR_GRAY2BGR) "
        f"if {in_var}.ndim == 2 else {in_var}"
    )


TRIANGLE_THRESHOLD = OperationSpec(
    id="threshold.triangle",
    name="Triangle Threshold",
    category="Threshold",
    description=(
        "Automatic threshold using the triangle method — particularly good "
        "for unimodal histograms where Otsu underperforms."
    ),
    parameters=(
        Parameter(name="maxval", kind="int", default=255, min=0, max=255, label="Max value"),
        Parameter(name="inverse", kind="bool", default=False, label="Invert"),
    ),
    func=_triangle_threshold,
    code_export=_triangle_code,
)


IN_RANGE_BGR = OperationSpec(
    id="threshold.in_range",
    name="In-Range (BGR)",
    category="Threshold",
    description=(
        "Returns a binary mask where every channel falls within the "
        "[low, high] interval. Useful for hand-tuned colour segmentation in "
        "BGR space (try the HSV in-range op for chromatic targeting)."
    ),
    parameters=(
        Parameter(name="b_low", kind="int", default=0, min=0, max=255, label="B low"),
        Parameter(name="b_high", kind="int", default=255, min=0, max=255, label="B high"),
        Parameter(name="g_low", kind="int", default=0, min=0, max=255, label="G low"),
        Parameter(name="g_high", kind="int", default=255, min=0, max=255, label="G high"),
        Parameter(name="r_low", kind="int", default=0, min=0, max=255, label="R low"),
        Parameter(name="r_high", kind="int", default=255, min=0, max=255, label="R high"),
    ),
    func=_in_range_bgr,
    code_export=_in_range_code,
)


ALL: tuple[OperationSpec, ...] = (
    BINARY_THRESHOLD,
    OTSU_THRESHOLD,
    ADAPTIVE_THRESHOLD,
    TRIANGLE_THRESHOLD,
    IN_RANGE_BGR,
)
