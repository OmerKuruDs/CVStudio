"""Color-space and channel operations.

These ops generally accept 3-channel BGR input. `to_grayscale` and `channel`
gracefully pass through if the image is already single-channel; `to_hsv` is
explicit about needing 3 channels.

`invert` is here rather than in filtering because it's the simplest color
transformation we have.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from cvsandbox.core.operation import OperationSpec, Parameter


def _to_grayscale_code(
    _params: dict[str, Any], input_vars: tuple[str, ...], output_var: str
) -> list[str]:
    (a,) = input_vars
    return [
        f"{output_var} = cv2.cvtColor({a}, cv2.COLOR_BGR2GRAY) "
        f"if {a}.ndim == 3 else {a}"
    ]


def _to_hsv_code(
    _params: dict[str, Any], input_vars: tuple[str, ...], output_var: str
) -> list[str]:
    (a,) = input_vars
    return [f"{output_var} = cv2.cvtColor({a}, cv2.COLOR_BGR2HSV)"]


def _invert_code(
    _params: dict[str, Any], input_vars: tuple[str, ...], output_var: str
) -> list[str]:
    (a,) = input_vars
    return [f"{output_var} = cv2.bitwise_not({a})"]


def _channel_code(
    params: dict[str, Any], input_vars: tuple[str, ...], output_var: str
) -> list[str]:
    (a,) = input_vars
    idx = int(params["channel"])
    return [
        f"{output_var} = {a} if {a}.ndim == 2 else {a}[:, :, {idx} % {a}.shape[2]]",
    ]


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _to_hsv(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("HSV conversion requires a 3-channel BGR image")
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def _invert(image: np.ndarray) -> np.ndarray:
    return cv2.bitwise_not(image)


def _channel(image: np.ndarray, channel: int) -> np.ndarray:
    if image.ndim == 2:
        return image  # nothing to extract
    idx = int(channel) % image.shape[2]
    return image[:, :, idx]


TO_GRAYSCALE = OperationSpec(
    id="color.to_grayscale",
    name="To Grayscale",
    category="Color",
    description="Converts BGR to single-channel grayscale. Pass-through if already grayscale.",
    parameters=(),
    func=_to_grayscale,
    code_export=_to_grayscale_code,
)


TO_HSV = OperationSpec(
    id="color.to_hsv",
    name="To HSV",
    category="Color",
    description="Converts BGR to HSV. Useful before thresholding on hue/saturation.",
    parameters=(),
    func=_to_hsv,
    code_export=_to_hsv_code,
)


INVERT = OperationSpec(
    id="color.invert",
    name="Invert",
    category="Color",
    description="255 - pixel. Works on any channel count.",
    parameters=(),
    func=_invert,
    code_export=_invert_code,
)


CHANNEL = OperationSpec(
    id="color.channel",
    name="Extract Channel",
    category="Color",
    description="Outputs a single channel by index (0/1/2 = B/G/R for BGR input, H/S/V for HSV).",
    parameters=(
        Parameter(
            name="channel",
            kind="int",
            default=0,
            min=0,
            max=2,
            label="Channel index",
        ),
    ),
    func=_channel,
    code_export=_channel_code,
)


def _clahe(image: np.ndarray, clip_limit: float, tile_grid: int) -> np.ndarray:
    grid = max(1, int(tile_grid))
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(grid, grid))
    if image.ndim == 2:
        return clahe.apply(image)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _clahe_code(
    params: dict[str, Any], input_vars: tuple[str, ...], output_var: str
) -> list[str]:
    (a,) = input_vars
    clip = float(params["clip_limit"])
    grid = max(1, int(params["tile_grid"]))
    return [
        f"_clahe = cv2.createCLAHE(clipLimit={clip}, tileGridSize=({grid}, {grid}))",
        f"if {a}.ndim == 2:",
        f"    {output_var} = _clahe.apply({a})",
        "else:",
        f"    _lab = cv2.cvtColor({a}, cv2.COLOR_BGR2LAB)",
        "    _lab[:, :, 0] = _clahe.apply(_lab[:, :, 0])",
        f"    {output_var} = cv2.cvtColor(_lab, cv2.COLOR_LAB2BGR)",
    ]


def _hsv_in_range(
    image: np.ndarray,
    h_min: int,
    h_max: int,
    s_min: int,
    s_max: int,
    v_min: int,
    v_max: int,
) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("HSV In-Range requires a 3-channel BGR image")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([int(h_min), int(s_min), int(v_min)], dtype=np.uint8)
    upper = np.array([int(h_max), int(s_max), int(v_max)], dtype=np.uint8)
    return cv2.inRange(hsv, lower, upper)


def _hsv_in_range_code(
    params: dict[str, Any], input_vars: tuple[str, ...], output_var: str
) -> list[str]:
    (a,) = input_vars
    return [
        f"_hsv = cv2.cvtColor({a}, cv2.COLOR_BGR2HSV)",
        f"_lower = np.array([{int(params['h_min'])}, {int(params['s_min'])}, "
        f"{int(params['v_min'])}], dtype=np.uint8)",
        f"_upper = np.array([{int(params['h_max'])}, {int(params['s_max'])}, "
        f"{int(params['v_max'])}], dtype=np.uint8)",
        f"{output_var} = cv2.inRange(_hsv, _lower, _upper)",
    ]


CLAHE = OperationSpec(
    id="color.clahe",
    name="CLAHE",
    category="Color",
    description="Contrast Limited Adaptive Histogram Equalization. Applied on the L channel for color images.",
    parameters=(
        Parameter(
            name="clip_limit",
            kind="float",
            default=2.0,
            min=0.1,
            max=20.0,
            step=0.1,
            label="Clip limit",
            description="Caps contrast amplification. Higher = more aggressive.",
        ),
        Parameter(
            name="tile_grid",
            kind="int",
            default=8,
            min=1,
            max=32,
            label="Tile grid",
            description="NxN tiles. Smaller = more local; larger = more global.",
        ),
    ),
    func=_clahe,
    code_export=_clahe_code,
)


HSV_IN_RANGE = OperationSpec(
    id="color.hsv_in_range",
    name="HSV In-Range Mask",
    category="Color",
    description="Binary mask of pixels whose HSV values fall in the given range. Input must be BGR.",
    parameters=(
        Parameter(name="h_min", kind="int", default=0, min=0, max=179, label="H min"),
        Parameter(name="h_max", kind="int", default=179, min=0, max=179, label="H max"),
        Parameter(name="s_min", kind="int", default=0, min=0, max=255, label="S min"),
        Parameter(name="s_max", kind="int", default=255, min=0, max=255, label="S max"),
        Parameter(name="v_min", kind="int", default=0, min=0, max=255, label="V min"),
        Parameter(name="v_max", kind="int", default=255, min=0, max=255, label="V max"),
    ),
    func=_hsv_in_range,
    code_export=_hsv_in_range_code,
)


ALL: tuple[OperationSpec, ...] = (TO_GRAYSCALE, TO_HSV, INVERT, CHANNEL, CLAHE, HSV_IN_RANGE)
