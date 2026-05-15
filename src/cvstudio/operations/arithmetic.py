"""Pixel arithmetic: add, subtract, multiply, bitwise operations.

All ops here are two-input (multi-port). Like the composite ops they auto-
match the second input's channel layout / size to the first via
``coerce_to_match``, so wiring a single-channel mask into a BGR pipeline
just works.

For weighted addition (``a*α + b*(1-α)``) reach for ``composite.blend``;
for absolute difference (``|a - b|``) use ``composite.difference``; for
``255 - image`` use ``color.invert`` — these were added earlier and the
arithmetic ops here intentionally avoid duplicating them.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from cvsandbox.core.operation import OperationSpec, Parameter
from cvsandbox.core.pipeline import coerce_to_match


def _match_to_a(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    matched = coerce_to_match(b, a)
    if matched.shape != a.shape:
        matched = cv2.resize(
            matched, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR
        )
    return matched


def _match_b_lines(a: str, b: str) -> list[str]:
    return [
        f"_b = _coerce_to_match({b}, {a})",
        f"if _b.shape != {a}.shape:",
        f"    _b = cv2.resize(_b, ({a}.shape[1], {a}.shape[0]), interpolation=cv2.INTER_LINEAR)",
    ]


# ----------------------------------------------------------------------- Add


def _add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return cv2.add(a, _match_to_a(a, b))


def _add_code(
    _params: dict[str, Any], input_vars: tuple[str, ...], output_var: str
) -> list[str]:
    a, b = input_vars
    return [*_match_b_lines(a, b), f"{output_var} = cv2.add({a}, _b)"]


ADD = OperationSpec(
    id="arithmetic.add",
    name="Add",
    category="Arithmetic",
    description=(
        "Saturated per-pixel sum (`a + b`, clipped to 255). Wire two images "
        "into <code>a</code> and <code>b</code> — channel layouts and sizes "
        "are auto-matched."
    ),
    parameters=(),
    func=_add,
    code_export=_add_code,
    input_ports=("a", "b"),
)


# ------------------------------------------------------------------ Subtract


def _subtract(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return cv2.subtract(a, _match_to_a(a, b))


def _subtract_code(
    _params: dict[str, Any], input_vars: tuple[str, ...], output_var: str
) -> list[str]:
    a, b = input_vars
    return [*_match_b_lines(a, b), f"{output_var} = cv2.subtract({a}, _b)"]


SUBTRACT = OperationSpec(
    id="arithmetic.subtract",
    name="Subtract",
    category="Arithmetic",
    description=(
        "Saturated per-pixel difference (`a - b`, clipped to 0). For unsigned "
        "absolute difference use <b>Difference</b> from the Composite category."
    ),
    parameters=(),
    func=_subtract,
    code_export=_subtract_code,
    input_ports=("a", "b"),
)


# ------------------------------------------------------------------ Multiply


def _multiply(a: np.ndarray, b: np.ndarray, scale: float) -> np.ndarray:
    return cv2.multiply(a, _match_to_a(a, b), scale=float(scale))


def _multiply_code(
    params: dict[str, Any], input_vars: tuple[str, ...], output_var: str
) -> list[str]:
    a, b = input_vars
    scale = float(params["scale"])
    return [
        *_match_b_lines(a, b),
        f"{output_var} = cv2.multiply({a}, _b, scale={scale})",
    ]


MULTIPLY = OperationSpec(
    id="arithmetic.multiply",
    name="Multiply",
    category="Arithmetic",
    description=(
        "Per-pixel product (`a · b · scale`, clipped to 255). With a binary "
        "mask on <code>b</code> + scale = 1/255 it works like Apply Mask but "
        "preserves grey levels for soft masks."
    ),
    parameters=(
        Parameter(
            name="scale",
            kind="float",
            default=1.0 / 255.0,
            min=0.0,
            max=1.0,
            step=0.001,
            label="Scale",
            description="Result is scaled by this. 1/255 ≈ 0.00392 keeps masked images in range.",
        ),
    ),
    func=_multiply,
    code_export=_multiply_code,
    input_ports=("a", "b"),
)


# --------------------------------------------------------------- Bitwise ops


def _bitwise_and(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return cv2.bitwise_and(a, _match_to_a(a, b))


def _bitwise_or(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return cv2.bitwise_or(a, _match_to_a(a, b))


def _bitwise_xor(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return cv2.bitwise_xor(a, _match_to_a(a, b))


def _bitwise_code_for(fn: str):
    def _code(
        _params: dict[str, Any], input_vars: tuple[str, ...], output_var: str
    ) -> list[str]:
        a, b = input_vars
        return [
            *_match_b_lines(a, b),
            f"{output_var} = cv2.{fn}({a}, _b)",
        ]

    return _code


BITWISE_AND = OperationSpec(
    id="arithmetic.bitwise_and",
    name="Bitwise AND",
    category="Arithmetic",
    description=(
        "Per-pixel bitwise AND. Most common use: combine two binary masks — "
        "the output is 255 only where both inputs are."
    ),
    parameters=(),
    func=_bitwise_and,
    code_export=_bitwise_code_for("bitwise_and"),
    input_ports=("a", "b"),
)


BITWISE_OR = OperationSpec(
    id="arithmetic.bitwise_or",
    name="Bitwise OR",
    category="Arithmetic",
    description=(
        "Per-pixel bitwise OR. With binary masks: the union — bright in either "
        "input becomes bright in the output."
    ),
    parameters=(),
    func=_bitwise_or,
    code_export=_bitwise_code_for("bitwise_or"),
    input_ports=("a", "b"),
)


BITWISE_XOR = OperationSpec(
    id="arithmetic.bitwise_xor",
    name="Bitwise XOR",
    category="Arithmetic",
    description=(
        "Per-pixel bitwise XOR. With binary masks: the symmetric difference — "
        "bright where exactly one of the two inputs is bright."
    ),
    parameters=(),
    func=_bitwise_xor,
    code_export=_bitwise_code_for("bitwise_xor"),
    input_ports=("a", "b"),
)


ALL: tuple[OperationSpec, ...] = (
    ADD,
    SUBTRACT,
    MULTIPLY,
    BITWISE_AND,
    BITWISE_OR,
    BITWISE_XOR,
)
