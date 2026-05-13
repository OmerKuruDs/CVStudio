"""Pipeline — ordered list of operations applied to an image.

A Pipeline holds a sequence of PipelineNodes. Each node binds an OperationSpec
to a concrete set of parameter values. Executing the pipeline copies the input
image once at the start, then folds each enabled node over it in order.

The original image is never mutated. Disabled nodes are skipped without affecting
downstream output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cvsandbox.core.operation import OperationSpec


@dataclass(slots=True)
class PipelineNode:
    spec: OperationSpec
    params: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

    def __post_init__(self) -> None:
        # Fill in defaults for unspecified params; reject unknown ones.
        defaults = self.spec.default_params()
        unknown = set(self.params) - set(defaults)
        if unknown:
            raise ValueError(f"Unknown parameter(s) for {self.spec.id}: {sorted(unknown)}")
        for name, default in defaults.items():
            self.params.setdefault(name, default)

    def execute(self, image: np.ndarray) -> np.ndarray:
        return self.spec.func(image, **self.params)


class Pipeline:
    def __init__(self) -> None:
        self._nodes: list[PipelineNode] = []

    @property
    def nodes(self) -> list[PipelineNode]:
        return self._nodes

    def __len__(self) -> int:
        return len(self._nodes)

    def add(self, spec: OperationSpec, params: dict[str, Any] | None = None) -> PipelineNode:
        node = PipelineNode(spec=spec, params=dict(params) if params else {})
        self._nodes.append(node)
        return node

    def remove(self, index: int) -> PipelineNode:
        return self._nodes.pop(index)

    def move(self, src: int, dst: int) -> None:
        node = self._nodes.pop(src)
        self._nodes.insert(dst, node)

    def clear(self) -> None:
        self._nodes.clear()

    def execute(self, image: np.ndarray) -> np.ndarray:
        current = image.copy()
        for node in self._nodes:
            if not node.enabled:
                continue
            current = node.execute(current)
        return current
