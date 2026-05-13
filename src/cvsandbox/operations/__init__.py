"""Built-in OpenCV operations.

Each submodule exposes an `ALL: tuple[OperationSpec, ...]` of the specs it
provides. `load_builtin_operations()` registers every built-in spec; it is
idempotent so it can safely be called multiple times (e.g. from tests).

We deliberately avoid registration as an import-time side effect: it makes
registration order implicit and makes the registry hard to reset in tests.
"""

from __future__ import annotations

from cvsandbox.core.operation import OperationSpec
from cvsandbox.core.registry import register_operation
from cvsandbox.operations import color, edge, filtering, geometric, morphology, threshold

_BUILTIN_MODULES = (filtering, threshold, morphology, edge, color, geometric)


def all_builtin_specs() -> tuple[OperationSpec, ...]:
    return tuple(spec for module in _BUILTIN_MODULES for spec in module.ALL)


def load_builtin_operations() -> None:
    """Register every built-in operation. Safe to call repeatedly."""
    from cvsandbox.core.registry import get_operation

    for spec in all_builtin_specs():
        try:
            get_operation(spec.id)
        except KeyError:
            register_operation(spec)


__all__ = ["all_builtin_specs", "load_builtin_operations"]
