"""Morphology operations: erode, dilate, open, close.

These accept any input shape (grayscale or color). The kernel is built from a
shape choice + odd kernel size. Iterations control how many times the op is
applied.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from cvstudio.core.operation import OperationSpec, Parameter

_SHAPES = {
    "Rectangle": cv2.MORPH_RECT,
    "Ellipse": cv2.MORPH_ELLIPSE,
    "Cross": cv2.MORPH_CROSS,
}

_SHAPE_CONSTS = {
    "Rectangle": "cv2.MORPH_RECT",
    "Ellipse": "cv2.MORPH_ELLIPSE",
    "Cross": "cv2.MORPH_CROSS",
}


def _kernel_line(params: dict[str, Any]) -> str:
    k = max(1, int(params["ksize"]) | 1)
    return (
        f"_kernel = cv2.getStructuringElement("
        f"{_SHAPE_CONSTS[params['shape']]}, ({k}, {k}))"
    )


def _erode_code(
    params: dict[str, Any], input_vars: tuple[str, ...], output_var: str
) -> list[str]:
    (a,) = input_vars
    return [
        _kernel_line(params),
        f"{output_var} = cv2.erode({a}, _kernel, iterations={int(params['iterations'])})",
    ]


def _dilate_code(
    params: dict[str, Any], input_vars: tuple[str, ...], output_var: str
) -> list[str]:
    (a,) = input_vars
    return [
        _kernel_line(params),
        f"{output_var} = cv2.dilate({a}, _kernel, iterations={int(params['iterations'])})",
    ]


def _morph_ex_code(
    op_const: str, params: dict[str, Any], input_var: str, output_var: str
) -> list[str]:
    return [
        _kernel_line(params),
        f"{output_var} = cv2.morphologyEx({input_var}, {op_const}, _kernel, "
        f"iterations={int(params['iterations'])})",
    ]


def _open_code(
    params: dict[str, Any], input_vars: tuple[str, ...], output_var: str
) -> list[str]:
    (a,) = input_vars
    return _morph_ex_code("cv2.MORPH_OPEN", params, a, output_var)


def _close_code(
    params: dict[str, Any], input_vars: tuple[str, ...], output_var: str
) -> list[str]:
    (a,) = input_vars
    return _morph_ex_code("cv2.MORPH_CLOSE", params, a, output_var)


def _kernel(shape: str, ksize: int) -> np.ndarray:
    k = max(1, int(ksize) | 1)
    return cv2.getStructuringElement(_SHAPES[shape], (k, k))


def _erode(image: np.ndarray, shape: str, ksize: int, iterations: int) -> np.ndarray:
    return cv2.erode(image, _kernel(shape, ksize), iterations=int(iterations))


def _dilate(image: np.ndarray, shape: str, ksize: int, iterations: int) -> np.ndarray:
    return cv2.dilate(image, _kernel(shape, ksize), iterations=int(iterations))


def _open(image: np.ndarray, shape: str, ksize: int, iterations: int) -> np.ndarray:
    return cv2.morphologyEx(
        image, cv2.MORPH_OPEN, _kernel(shape, ksize), iterations=int(iterations)
    )


def _close(image: np.ndarray, shape: str, ksize: int, iterations: int) -> np.ndarray:
    return cv2.morphologyEx(
        image, cv2.MORPH_CLOSE, _kernel(shape, ksize), iterations=int(iterations)
    )


def _shared_params() -> tuple[Parameter, ...]:
    return (
        Parameter(
            name="shape",
            kind="choice",
            default="Rectangle",
            choices=tuple(_SHAPES.keys()),
            label="Kernel shape",
        ),
        Parameter(
            name="ksize",
            kind="kernel_size",
            default=3,
            min=1,
            max=31,
            step=2,
            label="Kernel size",
        ),
        Parameter(
            name="iterations",
            kind="int",
            default=1,
            min=1,
            max=20,
            label="Iterations",
        ),
    )


ERODE = OperationSpec(
    id="morphology.erode",
    name="Erode",
    category="Morphology",
    description="Shrinks bright regions. Good for removing small bright noise.",
    parameters=_shared_params(),
    func=_erode,
    code_export=_erode_code,
)


DILATE = OperationSpec(
    id="morphology.dilate",
    name="Dilate",
    category="Morphology",
    description="Grows bright regions. Good for joining nearby bright blobs.",
    parameters=_shared_params(),
    func=_dilate,
    code_export=_dilate_code,
)


OPEN = OperationSpec(
    id="morphology.open",
    name="Open",
    category="Morphology",
    description="Erode then dilate. Removes small bright objects while preserving large shapes.",
    parameters=_shared_params(),
    func=_open,
    code_export=_open_code,
)


CLOSE = OperationSpec(
    id="morphology.close",
    name="Close",
    category="Morphology",
    description="Dilate then erode. Closes small holes in bright regions.",
    parameters=_shared_params(),
    func=_close,
    code_export=_close_code,
)


def _gradient(image: np.ndarray, shape: str, ksize: int, iterations: int) -> np.ndarray:
    return cv2.morphologyEx(
        image, cv2.MORPH_GRADIENT, _kernel(shape, ksize), iterations=int(iterations)
    )


def _tophat(image: np.ndarray, shape: str, ksize: int, iterations: int) -> np.ndarray:
    return cv2.morphologyEx(
        image, cv2.MORPH_TOPHAT, _kernel(shape, ksize), iterations=int(iterations)
    )


def _blackhat(image: np.ndarray, shape: str, ksize: int, iterations: int) -> np.ndarray:
    return cv2.morphologyEx(
        image, cv2.MORPH_BLACKHAT, _kernel(shape, ksize), iterations=int(iterations)
    )


def _gradient_code(
    params: dict[str, Any], input_vars: tuple[str, ...], output_var: str
) -> list[str]:
    (a,) = input_vars
    return _morph_ex_code("cv2.MORPH_GRADIENT", params, a, output_var)


def _tophat_code(
    params: dict[str, Any], input_vars: tuple[str, ...], output_var: str
) -> list[str]:
    (a,) = input_vars
    return _morph_ex_code("cv2.MORPH_TOPHAT", params, a, output_var)


def _blackhat_code(
    params: dict[str, Any], input_vars: tuple[str, ...], output_var: str
) -> list[str]:
    (a,) = input_vars
    return _morph_ex_code("cv2.MORPH_BLACKHAT", params, a, output_var)


GRADIENT = OperationSpec(
    id="morphology.gradient",
    name="Gradient",
    category="Morphology",
    description=(
        "Dilation minus erosion — produces an outline of bright regions. "
        "Cheap edge detector for binary or near-binary inputs."
    ),
    parameters=_shared_params(),
    func=_gradient,
    code_export=_gradient_code,
)


TOPHAT = OperationSpec(
    id="morphology.tophat",
    name="Top-Hat",
    category="Morphology",
    description=(
        "Input minus its opening — isolates bright features smaller than the "
        "kernel. Great for picking out small bright objects on a darker "
        "uneven background."
    ),
    parameters=_shared_params(),
    func=_tophat,
    code_export=_tophat_code,
)


BLACKHAT = OperationSpec(
    id="morphology.blackhat",
    name="Black-Hat",
    category="Morphology",
    description=(
        "Closing minus input — isolates dark features smaller than the "
        "kernel. The mirror of Top-Hat for dark blobs on a brighter "
        "background."
    ),
    parameters=_shared_params(),
    func=_blackhat,
    code_export=_blackhat_code,
)


ALL: tuple[OperationSpec, ...] = (
    ERODE,
    DILATE,
    OPEN,
    CLOSE,
    GRADIENT,
    TOPHAT,
    BLACKHAT,
)
