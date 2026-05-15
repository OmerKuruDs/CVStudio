from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cvstudio.core.operation import OperationSpec, Parameter
from cvstudio.core.pipeline import Pipeline, Roi
from cvstudio.core.registry import clear_registry, register_operation
from cvstudio.core.serialization import from_dict, load, save, to_dict


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
    assert data["version"] == 2
    assert len(data["nodes"]) == 1
    saved = data["nodes"][0]
    assert saved["spec_id"] == "serial_test.add"
    assert saved["params"] == {"value": 5}
    assert saved["enabled"] is False
    assert saved["position"] is None
    # Source → first chain node edge is auto-managed and round-trips.
    assert len(data["edges"]) == 1
    assert data["edges"][0]["source"] == pipe.source_node_id
    assert data["edges"][0]["target"] == node.id


def test_to_dict_captures_node_position_when_set() -> None:
    pipe = Pipeline()
    node = pipe.add(ADD, {"value": 1})
    node.position = (250.0, 90.0)
    data = to_dict(pipe)
    assert data["nodes"][0]["position"] == [250.0, 90.0]


def test_v2_round_trips_source_node_position() -> None:
    original = Pipeline()
    original.add(ADD, {"value": 1})
    original.graph.get_node(original.source_node_id).position = (50.0, 220.0)

    payload = to_dict(original)
    assert payload["source_position"] == [50.0, 220.0]

    restored = Pipeline()
    from_dict(payload, restored)
    assert restored.graph.get_node(restored.source_node_id).position == (50.0, 220.0)


def test_v2_round_trip_preserves_positions_and_extra_edges() -> None:
    original = Pipeline()
    a = original.add(ADD, {"value": 1})
    b = original.add(ADD, {"value": 2})
    a.position = (100.0, 20.0)
    b.position = (400.0, 200.0)

    restored = Pipeline()
    from_dict(to_dict(original), restored)

    assert restored.nodes[0].position == (100.0, 20.0)
    assert restored.nodes[1].position == (400.0, 200.0)
    # Chain edges (Source→a, a→b) survive identically.
    edge_pairs = {(e.source, e.target) for e in restored.graph.edges}
    assert (restored.source_node_id, restored.nodes[0].id) in edge_pairs
    assert (restored.nodes[0].id, restored.nodes[1].id) in edge_pairs


def test_v1_pipeline_file_is_auto_migrated_to_v2(tmp_path: Path) -> None:
    legacy = {
        "version": 1,
        "nodes": [
            {"id": "serial_test.add", "params": {"value": 5}, "enabled": True},
            {"id": "serial_test.add", "params": {"value": 7}, "enabled": False},
        ],
    }
    target = tmp_path / "legacy.cvpipe.json"
    target.write_text(json.dumps(legacy), encoding="utf-8")
    restored = Pipeline()
    load(target, restored)
    assert len(restored) == 2
    assert restored.nodes[0].params == {"value": 5}
    assert restored.nodes[1].params == {"value": 7}
    assert restored.nodes[1].enabled is False
    # Chain edges are derived from the linear v1 layout.
    edge_targets = {e.target for e in restored.graph.edges}
    assert restored.nodes[0].id in edge_targets
    assert restored.nodes[1].id in edge_targets


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


def test_roi_round_trips_through_to_dict_and_from_dict() -> None:
    original = Pipeline()
    original.add(ADD, {"value": 5})
    original.roi = Roi(x=10, y=20, width=30, height=40)

    payload = to_dict(original)
    assert payload["roi"] == {"x": 10, "y": 20, "width": 30, "height": 40}

    restored = Pipeline()
    from_dict(payload, restored)
    assert restored.roi == Roi(x=10, y=20, width=30, height=40)


def test_missing_roi_key_loads_as_none() -> None:
    restored = Pipeline()
    restored.roi = Roi(x=1, y=2, width=3, height=4)  # pre-existing — should be cleared
    from_dict({"version": 1, "nodes": []}, restored)
    assert restored.roi is None


def test_save_then_load_via_file(tmp_path: Path) -> None:
    original = Pipeline()
    original.add(ADD, {"value": 42})
    target = tmp_path / "test.cvpipe.json"
    save(original, target)

    assert target.exists()
    raw = json.loads(target.read_text(encoding="utf-8"))
    assert raw["version"] == 2
    assert raw["nodes"][0]["params"] == {"value": 42}

    restored = Pipeline()
    load(target, restored)
    assert restored.nodes[0].params == {"value": 42}


def test_v2_preserves_user_drawn_multi_input_edge() -> None:
    """An edge added directly through the Graph API (e.g. a drag-to-connect
    wire to a 'b' port) must round-trip — v1 could only express the chain."""
    from cvstudio.core.graph import GraphEdge
    from cvstudio.core.operation import OperationSpec as _OpSpec
    from cvstudio.core.operation import Parameter as _Param

    blend = _OpSpec(
        id="serial_test.blend",
        name="Blend",
        category="Test",
        description="",
        parameters=(_Param(name="alpha", kind="float", default=0.5, min=0.0, max=1.0),),
        func=lambda a, b, alpha: a,
        input_ports=("a", "b"),
    )
    register_operation(blend)

    original = Pipeline()
    a = original.add(ADD, {"value": 3})
    blend_node = original.add(blend, {"alpha": 0.5})
    # User wire from Source.image → blend.b (the second input port).
    original.graph.add_edge(
        GraphEdge(
            source=original.source_node_id,
            source_port="image",
            target=blend_node.id,
            target_port="b",
        )
    )
    payload = to_dict(original)

    restored = Pipeline()
    from_dict(payload, restored)
    edges = {(e.source, e.target, e.target_port) for e in restored.graph.edges}
    assert (restored.source_node_id, blend_node.id, "b") in edges
    # The auto-chain a→blend.a edge also survives.
    assert (a.id, blend_node.id, "a") in edges
