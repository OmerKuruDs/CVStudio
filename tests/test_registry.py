from __future__ import annotations

import numpy as np
import pytest

from cvstudio.core.operation import OperationSpec
from cvstudio.core.registry import (
    all_operations,
    clear_registry,
    get_operation,
    register_operation,
)


def _identity(image: np.ndarray) -> np.ndarray:
    return image


@pytest.fixture(autouse=True)
def _reset_registry() -> None:
    clear_registry()


def _make(spec_id: str, category: str = "Test", name: str = "X") -> OperationSpec:
    return OperationSpec(
        id=spec_id,
        name=name,
        category=category,
        description="",
        parameters=(),
        func=_identity,
    )


def test_register_and_get_roundtrip() -> None:
    spec = _make("filtering.x")
    register_operation(spec)
    assert get_operation("filtering.x") is spec


def test_register_rejects_duplicate_id() -> None:
    register_operation(_make("filtering.x"))
    with pytest.raises(ValueError, match="already registered"):
        register_operation(_make("filtering.x"))


def test_get_unknown_id_raises_keyerror() -> None:
    with pytest.raises(KeyError, match="Unknown operation id"):
        get_operation("does.not_exist")


def test_all_operations_sorted_by_category_then_name() -> None:
    register_operation(_make("morph.dilate", category="Morphology", name="Dilate"))
    register_operation(_make("morph.erode", category="Morphology", name="Erode"))
    register_operation(_make("filtering.blur", category="Filtering", name="Blur"))
    ops = all_operations()
    assert [op.id for op in ops] == ["filtering.blur", "morph.dilate", "morph.erode"]


def test_builtin_operations_register_on_load() -> None:
    from cvstudio.operations import load_builtin_operations

    load_builtin_operations()
    assert get_operation("filtering.gaussian_blur").name == "Gaussian Blur"
