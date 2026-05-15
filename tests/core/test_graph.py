from __future__ import annotations

import numpy as np
import pytest

from cvstudio.core.graph import Graph, GraphEdge, GraphNode
from cvstudio.core.operation import OperationSpec, Parameter


def _add_spec(spec_id: str = "test.add") -> OperationSpec:
    def _add(image: np.ndarray, value: int) -> np.ndarray:
        return np.clip(image.astype(np.int32) + value, 0, 255).astype(np.uint8)

    return OperationSpec(
        id=spec_id,
        name="Add",
        category="Test",
        description="",
        parameters=(Parameter(name="value", kind="int", default=10, min=-255, max=255),),
        func=_add,
    )


def _blend_spec() -> OperationSpec:
    """Two-input op: average two images element-wise."""

    def _blend(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
        return ((1.0 - alpha) * a + alpha * b).astype(np.uint8)

    return OperationSpec(
        id="test.blend",
        name="Blend",
        category="Test",
        description="",
        parameters=(
            Parameter(name="alpha", kind="float", default=0.5, min=0.0, max=1.0),
        ),
        func=_blend,
        input_ports=("a", "b"),
    )


def _split_channels_spec() -> OperationSpec:
    """Two-output op for testing multi-output mapping."""

    def _split(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if image.ndim < 3 or image.shape[2] < 2:
            return image, image
        return image[..., 0], image[..., 1]

    return OperationSpec(
        id="test.split",
        name="Split",
        category="Test",
        description="",
        parameters=(),
        func=_split,
        output_ports=("c0", "c1"),
    )


def _gray_image() -> np.ndarray:
    return np.full((4, 4, 3), 100, dtype=np.uint8)


# ---------------------------------------------------------------- basic shape


def test_empty_graph_passes_through_input() -> None:
    g = Graph()
    img = _gray_image()
    out = g.execute(img)
    assert np.array_equal(out, img)
    assert out is not img


def test_add_node_assigns_auto_id() -> None:
    g = Graph()
    n1 = g.add_node(_add_spec())
    n2 = g.add_node(_add_spec())
    assert n1.id != n2.id
    assert {n1.id, n2.id} == {n.id for n in g.nodes}


def test_add_node_with_explicit_id_rejects_duplicates() -> None:
    g = Graph()
    g.add_node(_add_spec(), node_id="alpha")
    with pytest.raises(ValueError, match="already in use"):
        g.add_node(_add_spec(), node_id="alpha")


def test_node_default_params_fill_in() -> None:
    g = Graph()
    node = g.add_node(_add_spec())
    assert node.params == {"value": 10}


def test_node_rejects_unknown_params() -> None:
    g = Graph()
    with pytest.raises(ValueError, match="Unknown parameter"):
        g.add_node(_add_spec(), params={"bogus": 1})


# ----------------------------------------------------------------------- edges


def test_add_edge_validates_node_existence() -> None:
    g = Graph()
    g.add_node(_add_spec(), node_id="a")
    with pytest.raises(ValueError, match="unknown target"):
        g.add_edge(GraphEdge(source="a", target="missing"))


def test_add_edge_rejects_unknown_port() -> None:
    g = Graph()
    g.add_node(_add_spec(), node_id="a")
    g.add_node(_add_spec(), node_id="b")
    with pytest.raises(ValueError, match="output port"):
        g.add_edge(GraphEdge(source="a", target="b", source_port="bogus"))


def test_add_edge_rejects_double_input_connection() -> None:
    g = Graph()
    g.add_node(_add_spec(), node_id="a")
    g.add_node(_add_spec(), node_id="b")
    g.add_node(_add_spec(), node_id="c")
    g.add_edge(GraphEdge(source="a", target="c"))
    with pytest.raises(ValueError, match="already connected"):
        g.add_edge(GraphEdge(source="b", target="c"))


def test_add_edge_rejects_cycle() -> None:
    g = Graph()
    g.add_node(_add_spec(), node_id="a")
    g.add_node(_add_spec(), node_id="b")
    g.add_edge(GraphEdge(source="a", target="b"))
    with pytest.raises(ValueError, match="cycle"):
        g.add_edge(GraphEdge(source="b", target="a"))


# -------------------------------------------------------------------- topology


def test_topological_order_is_deterministic_for_linear_chain() -> None:
    g = Graph()
    g.add_node(_add_spec(), node_id="a")
    g.add_node(_add_spec(), node_id="b")
    g.add_node(_add_spec(), node_id="c")
    g.add_edge(GraphEdge(source="a", target="b"))
    g.add_edge(GraphEdge(source="b", target="c"))
    assert g.topological_order() == ["a", "b", "c"]


def test_remove_node_drops_attached_edges() -> None:
    g = Graph()
    g.add_node(_add_spec(), node_id="a")
    g.add_node(_add_spec(), node_id="b")
    g.add_node(_add_spec(), node_id="c")
    g.add_edge(GraphEdge(source="a", target="b"))
    g.add_edge(GraphEdge(source="b", target="c"))
    g.remove_node("b")
    assert g.edges == []
    assert {n.id for n in g.nodes} == {"a", "c"}


# -------------------------------------------------------------------- execution


def test_linear_chain_runs_each_node_in_order() -> None:
    g = Graph()
    a = g.add_node(_add_spec(), params={"value": 10})
    b = g.add_node(_add_spec(), params={"value": 20})
    g.add_edge(GraphEdge(source=a.id, target=b.id))
    out = g.execute(_gray_image())  # 100 + 10 + 20 = 130
    assert int(out[0, 0, 0]) == 130


def test_disabled_node_is_passthrough() -> None:
    g = Graph()
    a = g.add_node(_add_spec(), params={"value": 50})
    b = g.add_node(_add_spec(), params={"value": 25})
    g.add_edge(GraphEdge(source=a.id, target=b.id))
    g._nodes[a.id].enabled = False
    out = g.execute(_gray_image())  # 100 + 25 (a skipped) = 125
    assert int(out[0, 0, 0]) == 125


def test_branching_fan_out_feeds_two_downstream_nodes() -> None:
    """One source → two downstream nodes share the same intermediate output."""
    g = Graph()
    src = g.add_node(_add_spec(), params={"value": 10})  # +10
    left = g.add_node(_add_spec(), params={"value": 20})  # +20
    right = g.add_node(_add_spec(), params={"value": 5})  # +5
    g.add_edge(GraphEdge(source=src.id, target=left.id))
    g.add_edge(GraphEdge(source=src.id, target=right.id))
    g.output_node_id = right.id
    out = g.execute(_gray_image())  # 100 + 10 + 5 = 115 (right is sink)
    assert int(out[0, 0, 0]) == 115


def test_merging_two_inputs_into_one_op() -> None:
    """Two upstream chains merge at a 2-input blend node."""
    g = Graph()
    a = g.add_node(_add_spec(), params={"value": 20})  # 100 + 20 = 120
    b = g.add_node(_add_spec(), params={"value": 60})  # 100 + 60 = 160
    blend = g.add_node(_blend_spec(), params={"alpha": 0.5})  # avg
    g.add_edge(GraphEdge(source=a.id, target=blend.id, target_port="a"))
    g.add_edge(GraphEdge(source=b.id, target=blend.id, target_port="b"))
    out = g.execute(_gray_image())  # (120 + 160) / 2 = 140
    assert int(out[0, 0, 0]) == 140


def test_multi_output_op_routes_each_port_independently() -> None:
    g = Graph()
    split = g.add_node(_split_channels_spec())
    follower = g.add_node(_add_spec(), params={"value": 0})
    g.add_edge(
        GraphEdge(source=split.id, target=follower.id, source_port="c1")
    )
    g.output_node_id = follower.id
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    img[..., 0] = 50  # channel 0
    img[..., 1] = 200  # channel 1
    out = g.execute(img)
    assert out.ndim == 2
    assert int(out[0, 0]) == 200


def test_unconnected_input_port_falls_back_to_graph_input() -> None:
    """A node whose input is not wired uses the original image — covers the
    'no incoming edge' branch in the execution loop."""
    g = Graph()
    node = g.add_node(_add_spec(), params={"value": 25})
    out = g.execute(_gray_image())
    assert int(out[0, 0, 0]) == 125
    assert node.id in g.topological_order()


def test_output_node_id_selects_the_sink() -> None:
    g = Graph()
    a = g.add_node(_add_spec(), params={"value": 10})
    b = g.add_node(_add_spec(), params={"value": 200})
    g.add_edge(GraphEdge(source=a.id, target=b.id))
    g.output_node_id = a.id  # ask for a's output, not the chain's tail
    out = g.execute(_gray_image())
    assert int(out[0, 0, 0]) == 110


# -------------------------------------------------------------------- defaults


def test_operationspec_default_ports() -> None:
    spec = _add_spec()
    assert spec.input_ports == ("in",)
    assert spec.output_ports == ("out",)


def test_graphnode_call_dispatches_positional_inputs() -> None:
    node = GraphNode(id="n", spec=_blend_spec(), params={"alpha": 0.25})
    a = np.full((2, 2), 0, dtype=np.uint8)
    b = np.full((2, 2), 100, dtype=np.uint8)
    out = node.call(a, b)
    assert int(out[0, 0]) == 25  # 0.75 * 0 + 0.25 * 100
