from __future__ import annotations

import numpy as np
import pytest

from cvsandbox.core.operation import OperationSpec, Parameter
from cvsandbox.core.pipeline import Pipeline, PipelineNode


def _add_constant(image: np.ndarray, value: int) -> np.ndarray:
    return np.clip(image.astype(np.int32) + value, 0, 255).astype(np.uint8)


ADD = OperationSpec(
    id="test.add_constant",
    name="Add Constant",
    category="Test",
    description="Adds a constant to every pixel.",
    parameters=(Parameter(name="value", kind="int", default=10, min=-255, max=255),),
    func=_add_constant,
)


def _gray_image() -> np.ndarray:
    return np.full((4, 4, 3), 100, dtype=np.uint8)


def test_empty_pipeline_returns_copy_of_input() -> None:
    pipe = Pipeline()
    img = _gray_image()
    out = pipe.execute(img)
    assert np.array_equal(out, img)
    assert out is not img, "pipeline must not return the input array itself"


def test_pipeline_applies_single_operation() -> None:
    pipe = Pipeline()
    pipe.add(ADD)
    out = pipe.execute(_gray_image())
    assert out[0, 0, 0] == 110


def test_pipeline_applies_operations_in_order() -> None:
    pipe = Pipeline()
    pipe.add(ADD, {"value": 10})
    pipe.add(ADD, {"value": 25})
    out = pipe.execute(_gray_image())
    assert out[0, 0, 0] == 135


def test_pipeline_node_fills_default_params() -> None:
    node = PipelineNode(spec=ADD)
    assert node.params == {"value": 10}


def test_pipeline_node_rejects_unknown_params() -> None:
    with pytest.raises(ValueError, match="Unknown parameter"):
        PipelineNode(spec=ADD, params={"bogus": 1})


def test_disabled_node_is_skipped() -> None:
    pipe = Pipeline()
    node = pipe.add(ADD, {"value": 50})
    node.enabled = False
    out = pipe.execute(_gray_image())
    assert out[0, 0, 0] == 100  # unchanged


def test_pipeline_remove_and_move() -> None:
    pipe = Pipeline()
    a = pipe.add(ADD, {"value": 1})
    b = pipe.add(ADD, {"value": 2})
    c = pipe.add(ADD, {"value": 3})
    pipe.move(0, 2)
    assert pipe.nodes == [b, c, a]
    removed = pipe.remove(1)
    assert removed is c
    assert len(pipe) == 2


def test_pipeline_does_not_mutate_input() -> None:
    pipe = Pipeline()
    pipe.add(ADD, {"value": 50})
    img = _gray_image()
    original = img.copy()
    pipe.execute(img)
    assert np.array_equal(img, original)
