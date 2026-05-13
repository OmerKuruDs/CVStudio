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

from cvsandbox.core.operation import OperationSpec, Parameter
from cvsandbox.core.pipeline import coerce_to_match


def _blend(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    b_matched = coerce_to_match(b, a)
    if a.shape != b_matched.shape:
        b_matched = cv2.resize(b_matched, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR)
    return cv2.addWeighted(a, 1.0 - float(alpha), b_matched, float(alpha), 0.0)


def _blend_code(params: dict[str, Any]) -> list[str]:
    alpha = float(params["alpha"])
    return [
        f"img = cv2.addWeighted(img, {1.0 - alpha}, img, {alpha}, 0.0)",
    ]


def _apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if mask.shape != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    binary = (mask > 0).astype(np.uint8) * 255
    return cv2.bitwise_and(image, image, mask=binary)


def _apply_mask_code(_params: dict[str, Any]) -> list[str]:
    return [
        "img = cv2.bitwise_and(img, img, mask=(img.any(axis=-1) * 255).astype('uint8') "
        "if img.ndim == 3 else (img > 0).astype('uint8') * 255)",
    ]


def _difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    b_matched = coerce_to_match(b, a)
    if a.shape != b_matched.shape:
        b_matched = cv2.resize(b_matched, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR)
    return cv2.absdiff(a, b_matched)


def _difference_code(_params: dict[str, Any]) -> list[str]:
    return ["img = cv2.absdiff(img, img)"]


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


ALL: tuple[OperationSpec, ...] = (BLEND, APPLY_MASK, DIFFERENCE)
