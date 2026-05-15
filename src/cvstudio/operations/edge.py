"""Edge detection operations: Canny, Sobel, Laplacian.

Canny and Laplacian want single-channel input; Sobel works per-channel but we
also coerce to gray for a consistent, edge-map-shaped output. Sobel and
Laplacian produce signed/float arrays that we collapse back to uint8 via
`convertScaleAbs` so the pipeline keeps a stable dtype.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from cvstudio.core.operation import OperationSpec, Parameter


def _to_gray_line(in_var: str, out_var: str) -> str:
    return (
        f"{out_var} = cv2.cvtColor({in_var}, cv2.COLOR_BGR2GRAY) "
        f"if {in_var}.ndim == 3 else {in_var}"
    )


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def _canny_code(
    params: dict[str, Any], input_vars: tuple[str, ...], output_var: str
) -> list[str]:
    (a,) = input_vars
    aperture = max(3, min(7, int(params["aperture_size"]) | 1))
    return [
        _to_gray_line(a, output_var),
        f"{output_var} = cv2.Canny({output_var}, {float(params['threshold1'])}, "
        f"{float(params['threshold2'])}, apertureSize={aperture})",
    ]


def _sobel_code(
    params: dict[str, Any], input_vars: tuple[str, ...], output_var: str
) -> list[str]:
    (a,) = input_vars
    dx = int(params["dx"])
    dy = int(params["dy"])
    if dx == 0 and dy == 0:
        return [_to_gray_line(a, output_var)]
    k = max(1, min(7, int(params["ksize"]) | 1))
    return [
        _to_gray_line(a, output_var),
        f"{output_var} = cv2.convertScaleAbs(cv2.Sobel({output_var}, cv2.CV_64F, "
        f"{dx}, {dy}, ksize={k}))",
    ]


def _laplacian_code(
    params: dict[str, Any], input_vars: tuple[str, ...], output_var: str
) -> list[str]:
    (a,) = input_vars
    k = max(1, min(31, int(params["ksize"]) | 1))
    return [
        _to_gray_line(a, output_var),
        f"{output_var} = cv2.convertScaleAbs(cv2.Laplacian({output_var}, "
        f"cv2.CV_64F, ksize={k}))",
    ]


def _canny(image: np.ndarray, threshold1: int, threshold2: int, aperture_size: int) -> np.ndarray:
    gray = _to_gray(image)
    aperture = max(3, min(7, int(aperture_size) | 1))
    return cv2.Canny(gray, float(threshold1), float(threshold2), apertureSize=aperture)


def _sobel(image: np.ndarray, dx: int, dy: int, ksize: int) -> np.ndarray:
    gray = _to_gray(image)
    # dx and dy can't both be zero — fall back to a no-op grayscale view.
    if int(dx) == 0 and int(dy) == 0:
        return gray
    k = max(1, min(7, int(ksize) | 1))
    raw = cv2.Sobel(gray, cv2.CV_64F, int(dx), int(dy), ksize=k)
    return cv2.convertScaleAbs(raw)


def _laplacian(image: np.ndarray, ksize: int) -> np.ndarray:
    gray = _to_gray(image)
    k = max(1, min(31, int(ksize) | 1))
    raw = cv2.Laplacian(gray, cv2.CV_64F, ksize=k)
    return cv2.convertScaleAbs(raw)


CANNY = OperationSpec(
    id="edge.canny",
    name="Canny",
    category="Edge",
    description="Canny edge detector. Outputs a binary edge map.",
    parameters=(
        Parameter(name="threshold1", kind="int", default=100, min=0, max=500, label="Threshold 1"),
        Parameter(name="threshold2", kind="int", default=200, min=0, max=500, label="Threshold 2"),
        Parameter(
            name="aperture_size",
            kind="kernel_size",
            default=3,
            min=3,
            max=7,
            step=2,
            label="Aperture size",
            description="Sobel kernel size used internally. Odd, 3-7.",
        ),
    ),
    func=_canny,
    code_export=_canny_code,
)


SOBEL = OperationSpec(
    id="edge.sobel",
    name="Sobel",
    category="Edge",
    description="First-order derivative. `dx`/`dy` choose the gradient direction.",
    parameters=(
        Parameter(name="dx", kind="int", default=1, min=0, max=2, label="dx"),
        Parameter(name="dy", kind="int", default=0, min=0, max=2, label="dy"),
        Parameter(
            name="ksize",
            kind="kernel_size",
            default=3,
            min=1,
            max=7,
            step=2,
            label="Kernel size",
        ),
    ),
    func=_sobel,
    code_export=_sobel_code,
)


LAPLACIAN = OperationSpec(
    id="edge.laplacian",
    name="Laplacian",
    category="Edge",
    description="Second-order derivative. Highlights regions of rapid intensity change.",
    parameters=(
        Parameter(
            name="ksize",
            kind="kernel_size",
            default=3,
            min=1,
            max=31,
            step=2,
            label="Kernel size",
        ),
    ),
    func=_laplacian,
    code_export=_laplacian_code,
)


def _scharr(image: np.ndarray, dx: int, dy: int) -> np.ndarray:
    gray = _to_gray(image)
    dx_i, dy_i = int(dx), int(dy)
    if dx_i == 0 and dy_i == 0:
        return gray
    if dx_i == 1 and dy_i == 1:
        # Scharr requires exactly one of dx/dy to be 1 — compose magnitude.
        gx = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
        gy = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
        return cv2.convertScaleAbs(np.abs(gx) + np.abs(gy))
    return cv2.convertScaleAbs(cv2.Scharr(gray, cv2.CV_64F, dx_i, dy_i))


def _scharr_code(
    params: dict[str, Any], input_vars: tuple[str, ...], output_var: str
) -> list[str]:
    (a,) = input_vars
    dx = int(params["dx"])
    dy = int(params["dy"])
    if dx == 0 and dy == 0:
        return [_to_gray_line(a, output_var)]
    if dx == 1 and dy == 1:
        return [
            _to_gray_line(a, "_gray"),
            "_gx = cv2.Scharr(_gray, cv2.CV_64F, 1, 0)",
            "_gy = cv2.Scharr(_gray, cv2.CV_64F, 0, 1)",
            f"{output_var} = cv2.convertScaleAbs(np.abs(_gx) + np.abs(_gy))",
        ]
    return [
        _to_gray_line(a, output_var),
        f"{output_var} = cv2.convertScaleAbs(cv2.Scharr({output_var}, "
        f"cv2.CV_64F, {dx}, {dy}))",
    ]


SCHARR = OperationSpec(
    id="edge.scharr",
    name="Scharr",
    category="Edge",
    description=(
        "Optimised 3×3 derivative kernel — sharper than Sobel for small "
        "kernels. Set both <i>dx</i> and <i>dy</i> to 1 for the magnitude of "
        "the gradient (|Gx| + |Gy|)."
    ),
    parameters=(
        Parameter(name="dx", kind="int", default=1, min=0, max=1, label="dx"),
        Parameter(name="dy", kind="int", default=0, min=0, max=1, label="dy"),
    ),
    func=_scharr,
    code_export=_scharr_code,
)


ALL: tuple[OperationSpec, ...] = (CANNY, SOBEL, LAPLACIAN, SCHARR)
