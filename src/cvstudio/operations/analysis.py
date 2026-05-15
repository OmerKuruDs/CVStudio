"""Analysis operations: contour detection, shape statistics.

These ops are most useful at the end of a pipeline (after thresholding) — they
draw visual overlays onto the image so the user can see what was detected.
Output is always a 3-channel BGR image so colored overlays survive.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from cvstudio.core.operation import OperationSpec, Parameter

_RETR_MODES = {
    "External": cv2.RETR_EXTERNAL,
    "All": cv2.RETR_LIST,
}

_RETR_CONSTS = {
    "External": "cv2.RETR_EXTERNAL",
    "All": "cv2.RETR_LIST",
}


def _to_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def _binary_for_contours(image: np.ndarray) -> np.ndarray:
    """Coerce input to a uint8 single-channel mask. Anything > 0 becomes 255."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    return mask


def _find_contours(
    image: np.ndarray,
    mode: str,
    min_area: int,
    thickness: int,
) -> np.ndarray:
    mask = _binary_for_contours(image)
    contours, _hierarchy = cv2.findContours(mask, _RETR_MODES[mode], cv2.CHAIN_APPROX_SIMPLE)
    if int(min_area) > 0:
        contours = tuple(c for c in contours if cv2.contourArea(c) >= float(min_area))
    canvas = _to_bgr(image).copy()
    cv2.drawContours(canvas, contours, -1, (0, 255, 0), int(thickness))
    return canvas


def _find_contours_code(
    params: dict[str, Any], input_vars: tuple[str, ...], output_var: str
) -> list[str]:
    (a,) = input_vars
    mode = _RETR_CONSTS[params["mode"]]
    min_area = int(params["min_area"])
    thickness = int(params["thickness"])
    return [
        f"_gray = {a} if {a}.ndim == 2 else cv2.cvtColor({a}, cv2.COLOR_BGR2GRAY)",
        "_, _mask = cv2.threshold(_gray, 0, 255, cv2.THRESH_BINARY)",
        f"_contours, _ = cv2.findContours(_mask, {mode}, cv2.CHAIN_APPROX_SIMPLE)",
        f"_contours = tuple(c for c in _contours if cv2.contourArea(c) >= {float(min_area)})"
        if min_area > 0
        else "# (no min-area filter)",
        f"_canvas = {a} if {a}.ndim == 3 else cv2.cvtColor({a}, cv2.COLOR_GRAY2BGR)",
        f"{output_var} = _canvas.copy()",
        f"cv2.drawContours({output_var}, _contours, -1, (0, 255, 0), {thickness})",
    ]


FIND_CONTOURS = OperationSpec(
    id="analysis.find_contours",
    name="Find Contours",
    category="Analysis",
    description=(
        "Detects contours on a binary input (any non-zero pixel counts as foreground) and "
        "draws them in green. Pair with a threshold op upstream."
    ),
    parameters=(
        Parameter(
            name="mode",
            kind="choice",
            default="External",
            choices=tuple(_RETR_MODES.keys()),
            label="Retrieval mode",
        ),
        Parameter(
            name="min_area",
            kind="int",
            default=0,
            min=0,
            max=100000,
            label="Min area (px²)",
            description="Discard contours smaller than this. 0 = keep all.",
        ),
        Parameter(
            name="thickness",
            kind="int",
            default=2,
            min=-1,
            max=10,
            label="Line thickness",
            description="-1 fills the contour interior.",
        ),
    ),
    func=_find_contours,
    code_export=_find_contours_code,
)


ALL: tuple[OperationSpec, ...] = (FIND_CONTOURS,)
