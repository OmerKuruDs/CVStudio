"""Geometric operations: resize, rotate, flip.

These preserve channel count and dtype. Rotation uses an affine warp with the
original image's bounding box, so corners that rotate outside the frame are
clipped (the canvas stays at the original size).
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from cvsandbox.core.operation import OperationSpec, Parameter

_INTERPOLATIONS = {
    "Nearest": cv2.INTER_NEAREST,
    "Linear": cv2.INTER_LINEAR,
    "Cubic": cv2.INTER_CUBIC,
    "Area": cv2.INTER_AREA,
}

_INTERPOLATION_CONSTS = {
    "Nearest": "cv2.INTER_NEAREST",
    "Linear": "cv2.INTER_LINEAR",
    "Cubic": "cv2.INTER_CUBIC",
    "Area": "cv2.INTER_AREA",
}

_FLIP_CODES = {
    "Horizontal": 1,
    "Vertical": 0,
    "Both": -1,
}


def _resize_code(
    params: dict[str, Any], input_vars: tuple[str, ...], output_var: str
) -> list[str]:
    (a,) = input_vars
    fx = max(0.01, float(params["scale_x"]))
    fy = max(0.01, float(params["scale_y"]))
    interp = _INTERPOLATION_CONSTS[params["interpolation"]]
    return [
        f"{output_var} = cv2.resize({a}, None, fx={fx}, fy={fy}, interpolation={interp})"
    ]


def _rotate_code(
    params: dict[str, Any], input_vars: tuple[str, ...], output_var: str
) -> list[str]:
    (a,) = input_vars
    angle = float(params["angle"])
    return [
        f"_h, _w = {a}.shape[:2]",
        f"_matrix = cv2.getRotationMatrix2D((_w / 2.0, _h / 2.0), {angle}, 1.0)",
        f"{output_var} = cv2.warpAffine({a}, _matrix, (_w, _h))",
    ]


def _flip_code(
    params: dict[str, Any], input_vars: tuple[str, ...], output_var: str
) -> list[str]:
    (a,) = input_vars
    code = _FLIP_CODES[params["mode"]]
    return [f"{output_var} = cv2.flip({a}, {code})"]


def _resize(image: np.ndarray, scale_x: float, scale_y: float, interpolation: str) -> np.ndarray:
    fx = max(0.01, float(scale_x))
    fy = max(0.01, float(scale_y))
    return cv2.resize(image, None, fx=fx, fy=fy, interpolation=_INTERPOLATIONS[interpolation])


def _rotate(image: np.ndarray, angle: float) -> np.ndarray:
    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, float(angle), 1.0)
    return cv2.warpAffine(image, matrix, (width, height))


def _flip(image: np.ndarray, mode: str) -> np.ndarray:
    return cv2.flip(image, _FLIP_CODES[mode])


RESIZE = OperationSpec(
    id="geometric.resize",
    name="Resize",
    category="Geometric",
    description="Scales the image. Independent X and Y factors; pick an interpolation method.",
    parameters=(
        Parameter(
            name="scale_x",
            kind="float",
            default=1.0,
            min=0.05,
            max=5.0,
            step=0.05,
            label="Scale X",
        ),
        Parameter(
            name="scale_y",
            kind="float",
            default=1.0,
            min=0.05,
            max=5.0,
            step=0.05,
            label="Scale Y",
        ),
        Parameter(
            name="interpolation",
            kind="choice",
            default="Linear",
            choices=tuple(_INTERPOLATIONS.keys()),
            label="Interpolation",
        ),
    ),
    func=_resize,
    code_export=_resize_code,
)


ROTATE = OperationSpec(
    id="geometric.rotate",
    name="Rotate",
    category="Geometric",
    description="Rotates around the image center by `angle` degrees. Canvas size is preserved.",
    parameters=(
        Parameter(
            name="angle",
            kind="float",
            default=0.0,
            min=-360.0,
            max=360.0,
            step=1.0,
            label="Angle (deg)",
        ),
    ),
    func=_rotate,
    code_export=_rotate_code,
)


FLIP = OperationSpec(
    id="geometric.flip",
    name="Flip",
    category="Geometric",
    description="Mirrors the image horizontally, vertically, or both.",
    parameters=(
        Parameter(
            name="mode",
            kind="choice",
            default="Horizontal",
            choices=tuple(_FLIP_CODES.keys()),
            label="Mode",
        ),
    ),
    func=_flip,
    code_export=_flip_code,
)


ALL: tuple[OperationSpec, ...] = (RESIZE, ROTATE, FLIP)
