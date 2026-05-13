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

ParamKind = Literal["int", "float", "bool", "choice", "kernel_size"]

OperationFunc = Callable[..., np.ndarray]


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
