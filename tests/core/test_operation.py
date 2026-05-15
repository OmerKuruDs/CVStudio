from __future__ import annotations

import numpy as np
import pytest

from cvstudio.core.operation import OperationSpec, Parameter


def _identity(image: np.ndarray) -> np.ndarray:
    return image


def test_parameter_int_requires_bounds() -> None:
    with pytest.raises(ValueError, match="requires min and max"):
        Parameter(name="x", kind="int", default=0)


def test_parameter_choice_requires_choices() -> None:
    with pytest.raises(ValueError, match="requires choices"):
        Parameter(name="mode", kind="choice", default="a")


def test_parameter_min_greater_than_max_rejected() -> None:
    with pytest.raises(ValueError, match=r"min .* > max"):
        Parameter(name="x", kind="int", default=0, min=10, max=5)


def test_parameter_bool_does_not_require_bounds() -> None:
    p = Parameter(name="flag", kind="bool", default=True)
    assert p.default is True


def test_parameter_display_label_falls_back_to_name() -> None:
    p = Parameter(name="ksize", kind="int", default=1, min=1, max=10)
    assert p.display_label == "ksize"
    p2 = Parameter(name="ksize", kind="int", default=1, min=1, max=10, label="Kernel size")
    assert p2.display_label == "Kernel size"


def test_operation_spec_id_must_have_category_prefix() -> None:
    with pytest.raises(ValueError, match=r"<category>\.<name>"):
        OperationSpec(
            id="no_dot",
            name="X",
            category="Misc",
            description="",
            parameters=(),
            func=_identity,
        )


def test_operation_spec_rejects_duplicate_parameter_names() -> None:
    p = Parameter(name="ksize", kind="int", default=1, min=1, max=10)
    with pytest.raises(ValueError, match="Duplicate parameter"):
        OperationSpec(
            id="filtering.x",
            name="X",
            category="Filtering",
            description="",
            parameters=(p, p),
            func=_identity,
        )


def test_operation_spec_default_params() -> None:
    spec = OperationSpec(
        id="filtering.x",
        name="X",
        category="Filtering",
        description="",
        parameters=(
            Parameter(name="a", kind="int", default=3, min=1, max=10),
            Parameter(name="b", kind="float", default=0.5, min=0.0, max=1.0),
        ),
        func=_identity,
    )
    assert spec.default_params() == {"a": 3, "b": 0.5}
