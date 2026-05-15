"""OperationSpec and Parameter — declarative description of an OpenCV operation.

An operation is a pure function `(image, **params) -> image` plus a Parameter list
that lets the UI auto-generate sliders/inputs. Specs are intentionally serializable
data (frozen dataclasses) so pipelines can be saved/loaded as JSON later.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

ParamKind = Literal["int", "float", "bool", "choice", "kernel_size", "string"]

OperationFunc = Callable[..., np.ndarray]
CodeExporter = Callable[[dict[str, Any], tuple[str, ...], str], list[str]]
"""Returns the Python source lines that reproduce this op for given params.

Signature: ``code_export(params, input_vars, output_var) -> list[str]``.

* ``params`` — the same dict ``func`` would receive at runtime.
* ``input_vars`` — variable names already holding each upstream input, in the
  positional order ``spec.input_ports`` declares. A single-input op gets a
  one-tuple; a multi-input op (Blend, Apply Mask, ...) gets one entry per port.
* ``output_var`` — the variable the lines must assign the result to so
  downstream steps can refer to it.

The lines must be self-contained — no shared helpers beyond ``cv2``, ``np``,
and short ``_``-prefixed local temporaries reset per node. The exporter bakes
literal values (post-clamping, post-odd-forcing) so the generated code mirrors
what ``func`` does at runtime.
"""


@dataclass(frozen=True, slots=True)
class Parameter:
    name: str
    kind: ParamKind
    default: Any
    min: float | int | None = None
    max: float | int | None = None
    step: float | int | None = None
    choices: tuple[str, ...] | None = None
    label: str | None = None
    description: str = ""

    def __post_init__(self) -> None:
        if self.kind in ("int", "float", "kernel_size"):
            if self.min is None or self.max is None:
                raise ValueError(
                    f"Parameter '{self.name}' of kind '{self.kind}' requires min and max"
                )
            if self.min > self.max:
                raise ValueError(f"Parameter '{self.name}': min ({self.min}) > max ({self.max})")
        if self.kind == "choice" and not self.choices:
            raise ValueError(f"Parameter '{self.name}' of kind 'choice' requires choices")

    @property
    def display_label(self) -> str:
        return self.label or self.name


@dataclass(frozen=True, slots=True)
class OperationSpec:
    id: str
    name: str
    category: str
    description: str
    parameters: tuple[Parameter, ...]
    func: OperationFunc
    code_export: CodeExporter | None = None
    input_ports: tuple[str, ...] = ("in",)
    """Names of the inputs the op accepts, in the positional order `func`
    expects. Default = a single image input named "in" (all 24 built-in
    operations). Multi-input ops (blend, mask, ...) override this."""
    output_ports: tuple[str, ...] = ("out",)
    """Names of the outputs the op produces. Default = a single image named
    "out". Multi-output ops (channel split, ...) override this."""
    manual_trigger: bool = False
    """When True, the op declares that it should not auto-run on every
    param change — `func` is expected to render a placeholder until the
    user explicitly requests execution (e.g. by clicking a Run button in
    the parameter panel). Used by expensive ops like VLM queries where
    each invocation costs seconds and external resources."""

    def __post_init__(self) -> None:
        if not self.id or "." not in self.id:
            raise ValueError(
                f"OperationSpec id must be in '<category>.<name>' form, got: {self.id!r}"
            )
        seen: set[str] = set()
        for p in self.parameters:
            if p.name in seen:
                raise ValueError(f"Duplicate parameter name in {self.id}: {p.name}")
            seen.add(p.name)

    def default_params(self) -> dict[str, Any]:
        return {p.name: p.default for p in self.parameters}
