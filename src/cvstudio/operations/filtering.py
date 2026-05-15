"""Filtering operations: smoothing, denoising."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from cvstudio.core.operation import OperationSpec, Parameter


def _gaussian_blur(image: np.ndarray, ksize: int, sigma_x: float) -> np.ndarray:
    k = int(ksize) | 1  # cv2 requires odd kernel sizes
    return cv2.GaussianBlur(image, (k, k), float(sigma_x))


def _gaussian_blur_code(
    params: dict[str, Any], input_vars: tuple[str, ...], output_var: str
) -> list[str]:
    (a,) = input_vars
    k = int(params["ksize"]) | 1
    sigma = float(params["sigma_x"])
    return [f"{output_var} = cv2.GaussianBlur({a}, ({k}, {k}), {sigma})"]


def _median_blur(image: np.ndarray, ksize: int) -> np.ndarray:
    return cv2.medianBlur(image, int(ksize) | 1)


def _median_blur_code(
    params: dict[str, Any], input_vars: tuple[str, ...], output_var: str
) -> list[str]:
    (a,) = input_vars
    k = int(params["ksize"]) | 1
    return [f"{output_var} = cv2.medianBlur({a}, {k})"]


GAUSSIAN_BLUR = OperationSpec(
    id="filtering.gaussian_blur",
    name="Gaussian Blur",
    category="Filtering",
    description="Smooths the image with a Gaussian kernel.",
    parameters=(
        Parameter(
            name="ksize",
            kind="kernel_size",
            default=3,
            min=1,
            max=99,
            step=2,
            label="Kernel size",
            description="Odd integer; larger = blurrier.",
        ),
        Parameter(
            name="sigma_x",
            kind="float",
            default=0.0,
            min=0.0,
            max=20.0,
            step=0.1,
            label="Sigma X",
            description="Gaussian standard deviation. 0 = derive from ksize.",
        ),
    ),
    func=_gaussian_blur,
    code_export=_gaussian_blur_code,
)


MEDIAN_BLUR = OperationSpec(
    id="filtering.median_blur",
    name="Median Blur",
    category="Filtering",
    description="Replaces each pixel with the median of its neighborhood. Strong against salt-and-pepper noise.",
    parameters=(
        Parameter(
            name="ksize",
            kind="kernel_size",
            default=3,
            min=1,
            max=99,
            step=2,
            label="Kernel size",
            description="Odd integer; larger = stronger denoising.",
        ),
    ),
    func=_median_blur,
    code_export=_median_blur_code,
)


def _bilateral(image: np.ndarray, d: int, sigma_color: float, sigma_space: float) -> np.ndarray:
    return cv2.bilateralFilter(image, int(d), float(sigma_color), float(sigma_space))


def _bilateral_code(
    params: dict[str, Any], input_vars: tuple[str, ...], output_var: str
) -> list[str]:
    (a,) = input_vars
    return [
        f"{output_var} = cv2.bilateralFilter({a}, {int(params['d'])}, "
        f"{float(params['sigma_color'])}, {float(params['sigma_space'])})"
    ]


def _nl_means(image: np.ndarray, strength: float, template_size: int, search_size: int) -> np.ndarray:
    t = max(3, int(template_size) | 1)
    s = max(t + 2, int(search_size) | 1)
    if image.ndim == 3:
        return cv2.fastNlMeansDenoisingColored(image, None, float(strength), float(strength), t, s)
    return cv2.fastNlMeansDenoising(image, None, float(strength), t, s)


def _nl_means_code(
    params: dict[str, Any], input_vars: tuple[str, ...], output_var: str
) -> list[str]:
    (a,) = input_vars
    strength = float(params["strength"])
    t = max(3, int(params["template_size"]) | 1)
    s = max(t + 2, int(params["search_size"]) | 1)
    return [
        f"if {a}.ndim == 3:",
        f"    {output_var} = cv2.fastNlMeansDenoisingColored({a}, None, {strength}, {strength}, {t}, {s})",
        "else:",
        f"    {output_var} = cv2.fastNlMeansDenoising({a}, None, {strength}, {t}, {s})",
    ]


BILATERAL = OperationSpec(
    id="filtering.bilateral",
    name="Bilateral Filter",
    category="Filtering",
    description="Edge-preserving smoothing. Slower than Gaussian; keeps strong edges sharp.",
    parameters=(
        Parameter(
            name="d",
            kind="int",
            default=9,
            min=1,
            max=25,
            label="Diameter",
            description="Pixel neighborhood diameter. 5 = fast, 9 = balanced, larger = very slow.",
        ),
        Parameter(
            name="sigma_color",
            kind="float",
            default=75.0,
            min=1.0,
            max=200.0,
            step=1.0,
            label="Sigma color",
            description="Higher = colors farther apart get mixed (more smoothing across edges).",
        ),
        Parameter(
            name="sigma_space",
            kind="float",
            default=75.0,
            min=1.0,
            max=200.0,
            step=1.0,
            label="Sigma space",
            description="Higher = farther pixels influence each other (more global smoothing).",
        ),
    ),
    func=_bilateral,
    code_export=_bilateral_code,
)


NL_MEANS = OperationSpec(
    id="filtering.nl_means",
    name="NL-Means Denoise",
    category="Filtering",
    description="Non-local means denoising. Slow but very effective on Gaussian noise.",
    parameters=(
        Parameter(
            name="strength",
            kind="float",
            default=10.0,
            min=1.0,
            max=50.0,
            step=1.0,
            label="Strength (h)",
            description="Filter strength. Higher = more denoising and more loss of detail.",
        ),
        Parameter(
            name="template_size",
            kind="kernel_size",
            default=7,
            min=3,
            max=21,
            step=2,
            label="Template size",
            description="Odd, ≥3. Size of the patch used to compare pixels.",
        ),
        Parameter(
            name="search_size",
            kind="kernel_size",
            default=21,
            min=5,
            max=51,
            step=2,
            label="Search size",
            description="Odd. Size of the search window. Bigger = slower.",
        ),
    ),
    func=_nl_means,
    code_export=_nl_means_code,
)


def _unsharp_mask(
    image: np.ndarray, radius: int, amount: float, threshold: int
) -> np.ndarray:
    k = max(1, int(radius) | 1)
    blurred = cv2.GaussianBlur(image, (k, k), 0)
    diff = image.astype(np.int16) - blurred.astype(np.int16)
    if int(threshold) > 0:
        diff = np.where(np.abs(diff) >= int(threshold), diff, 0)
    out = image.astype(np.float32) + float(amount) * diff.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def _unsharp_mask_code(
    params: dict[str, Any], input_vars: tuple[str, ...], output_var: str
) -> list[str]:
    (a,) = input_vars
    k = max(1, int(params["radius"]) | 1)
    amount = float(params["amount"])
    thr = int(params["threshold"])
    return [
        f"_blurred = cv2.GaussianBlur({a}, ({k}, {k}), 0)",
        f"_diff = {a}.astype(np.int16) - _blurred.astype(np.int16)",
        f"_diff = np.where(np.abs(_diff) >= {thr}, _diff, 0)" if thr > 0 else "# (no threshold)",
        f"_out = {a}.astype(np.float32) + {amount} * _diff.astype(np.float32)",
        f"{output_var} = np.clip(_out, 0, 255).astype(np.uint8)",
    ]


UNSHARP_MASK = OperationSpec(
    id="filtering.unsharp_mask",
    name="Unsharp Mask",
    category="Filtering",
    description=(
        "Classic photographic sharpening: subtract a blurred copy from the "
        "original to isolate detail, then add it back multiplied by "
        "<i>amount</i>. <i>Threshold</i> protects flat areas from being sharpened."
    ),
    parameters=(
        Parameter(
            name="radius",
            kind="kernel_size",
            default=3,
            min=1,
            max=31,
            step=2,
            label="Blur radius (px)",
            description="Odd kernel size for the Gaussian copy.",
        ),
        Parameter(
            name="amount",
            kind="float",
            default=1.0,
            min=0.0,
            max=5.0,
            step=0.1,
            label="Amount",
            description="How strongly to boost detail. 0 = no change, 1-2 = typical.",
        ),
        Parameter(
            name="threshold",
            kind="int",
            default=0,
            min=0,
            max=128,
            label="Threshold",
            description="Only sharpen pixels whose detail magnitude exceeds this. "
            "0 = sharpen everything.",
        ),
    ),
    func=_unsharp_mask,
    code_export=_unsharp_mask_code,
)


_KERNEL_PRESETS: dict[str, tuple[tuple[float, ...], ...]] = {
    "Identity": ((0, 0, 0), (0, 1, 0), (0, 0, 0)),
    "Sharpen": ((0, -1, 0), (-1, 5, -1), (0, -1, 0)),
    "Strong Sharpen": ((-1, -1, -1), (-1, 9, -1), (-1, -1, -1)),
    "Edge Enhance": ((0, -1, 0), (-1, 4, -1), (0, -1, 0)),
    "Emboss": ((-2, -1, 0), (-1, 1, 1), (0, 1, 2)),
    "Outline": ((-1, -1, -1), (-1, 8, -1), (-1, -1, -1)),
    "Box Blur 3x3": ((1 / 9, 1 / 9, 1 / 9), (1 / 9, 1 / 9, 1 / 9), (1 / 9, 1 / 9, 1 / 9)),
}


def _custom_kernel(image: np.ndarray, preset: str, strength: float) -> np.ndarray:
    base = np.array(_KERNEL_PRESETS[preset], dtype=np.float32)
    kernel = base * float(strength)
    out = cv2.filter2D(image, ddepth=cv2.CV_32F, kernel=kernel)
    return np.clip(out, 0, 255).astype(np.uint8)


def _custom_kernel_code(
    params: dict[str, Any], input_vars: tuple[str, ...], output_var: str
) -> list[str]:
    (a,) = input_vars
    preset = params["preset"]
    base = _KERNEL_PRESETS[preset]
    strength = float(params["strength"])
    rows = ", ".join(
        "(" + ", ".join(f"{v * strength:.6g}" for v in row) + ")" for row in base
    )
    return [
        f"_kernel = np.array(({rows}), dtype=np.float32)",
        f"_out = cv2.filter2D({a}, ddepth=cv2.CV_32F, kernel=_kernel)",
        f"{output_var} = np.clip(_out, 0, 255).astype(np.uint8)",
    ]


CUSTOM_KERNEL = OperationSpec(
    id="filtering.custom_kernel",
    name="Custom Kernel",
    category="Filtering",
    description=(
        "Applies a 3×3 convolution chosen from a preset list. <i>Strength</i> "
        "scales the whole kernel so you can dial the effect in or out."
    ),
    parameters=(
        Parameter(
            name="preset",
            kind="choice",
            default="Sharpen",
            choices=tuple(_KERNEL_PRESETS.keys()),
            label="Kernel preset",
        ),
        Parameter(
            name="strength",
            kind="float",
            default=1.0,
            min=0.0,
            max=3.0,
            step=0.05,
            label="Strength",
        ),
    ),
    func=_custom_kernel,
    code_export=_custom_kernel_code,
)


ALL: tuple[OperationSpec, ...] = (
    GAUSSIAN_BLUR,
    MEDIAN_BLUR,
    BILATERAL,
    NL_MEANS,
    UNSHARP_MASK,
    CUSTOM_KERNEL,
)
