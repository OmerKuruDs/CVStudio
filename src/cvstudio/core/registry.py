"""Global operation registry.

Operations register themselves at import time via `register_operation`. The UI
and pipeline serialization layer look up specs by id through this module.
"""

from __future__ import annotations

from cvstudio.core.operation import OperationSpec

_REGISTRY: dict[str, OperationSpec] = {}


def register_operation(spec: OperationSpec) -> None:
    if spec.id in _REGISTRY:
        raise ValueError(f"Operation id already registered: {spec.id}")
    _REGISTRY[spec.id] = spec


def get_operation(spec_id: str) -> OperationSpec:
    try:
        return _REGISTRY[spec_id]
    except KeyError as e:
        raise KeyError(f"Unknown operation id: {spec_id}") from e


def all_operations() -> list[OperationSpec]:
    return sorted(_REGISTRY.values(), key=lambda op: (op.category, op.name))


def clear_registry() -> None:
    """Reset the registry. Intended for tests only."""
    _REGISTRY.clear()
