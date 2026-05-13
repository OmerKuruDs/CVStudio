from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cvsandbox.core.operation import OperationSpec, Parameter
from cvsandbox.core.pipeline import Pipeline
from cvsandbox.core.registry import clear_registry, register_operation
from cvsandbox.core.serialization import from_dict, load, save, to_dict


def _add(image: np.ndarray, value: int) -> np.ndarray:
    return image + value


ADD = OperationSpec(
    id="serial_test.add",
    name="Add",
    category="Test",
    description="Adds a constant.",
    parameters=(Parameter(name="value", kind="int", default=0, min=-100, max=100),),
    func=_add,
)


@pytest.fixture(autouse=True)
def _register_test_op() -> None:
    clear_registry()
    register_operation(ADD)


def test_to_dict_captures_id_params_enabled() -> None:
    pipe = Pipeline()
    node = pipe.add(ADD, {"value": 5})
    node.enabled = False
    data = to_dict(pipe)
    assert data == {
        "version": 1,
        "nodes": [{"id": "serial_test.add", "params": {"value": 5}, "enabled": False}],
    }


def test_from_dict_round_trip() -> None:
    original = Pipeline()
    original.add(ADD, {"value": 3})
    original.add(ADD, {"value": 7})

    restored = Pipeline()
    from_dict(to_dict(original), restored)

    assert len(restored) == 2
    assert restored.nodes[0].params == {"value": 3}
    assert restored.nodes[1].params == {"value": 7}


def test_from_dict_rejects_unknown_version() -> None:
    pipe = Pipeline()
    with pytest.raises(ValueError, match="Unsupported pipeline version"):
        from_dict({"version": 99, "nodes": []}, pipe)


def test_from_dict_is_atomic_on_unknown_operation() -> None:
    pipe = Pipeline()
    pipe.add(ADD, {"value": 1})  # existing content we must NOT lose on failure
    bad_data = {
        "version": 1,
        "nodes": [
            {"id": "serial_test.add", "params": {"value": 99}, "enabled": True},
            {"id": "does.not.exist", "params": {}, "enabled": True},
        ],
    }
    with pytest.raises(KeyError):
        from_dict(bad_data, pipe)
    assert len(pipe) == 1
    assert pipe.nodes[0].params == {"value": 1}


def test_save_then_load_via_file(tmp_path: Path) -> None:
    original = Pipeline()
    original.add(ADD, {"value": 42})
    target = tmp_path / "test.cvpipe.json"
    save(original, target)

    assert target.exists()
    raw = json.loads(target.read_text(encoding="utf-8"))
    assert raw["version"] == 1
    assert raw["nodes"][0]["params"] == {"value": 42}

    restored = Pipeline()
    load(target, restored)
    assert restored.nodes[0].params == {"value": 42}
