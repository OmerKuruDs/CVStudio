from __future__ import annotations

import pytest
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication

from cvsandbox.core.operation import OperationSpec, Parameter
from cvsandbox.core.pipeline import Pipeline
from cvsandbox.ui.node_graph_view import (
    NODE_GAP,
    NODE_WIDTH,
    SCENE_MARGIN,
    NodeGraphView,
    _layout_x,
    format_duration,
)


def _spec(spec_id: str, name: str, category: str = "Test") -> OperationSpec:
    return OperationSpec(
        id=spec_id,
        name=name,
        category=category,
        description="",
        parameters=(Parameter(name="x", kind="int", default=0, min=0, max=10),),
        func=lambda image, x: image,
    )


@pytest.fixture
def view(qapp: QApplication) -> NodeGraphView:
    pipeline = Pipeline()
    pipeline.add(_spec("test.a", "Alpha"))
    pipeline.add(_spec("test.b", "Beta"))
    pipeline.add(_spec("test.c", "Gamma"))
    widget = NodeGraphView(pipeline)
    widget.resize(800, 200)
    return widget


# --------------------------------------------------------------- format_duration


def test_format_duration_microseconds() -> None:
    assert format_duration(0.000_25) == "250 µs"


def test_format_duration_milliseconds() -> None:
    assert format_duration(0.012_3) == "12.3 ms"


def test_format_duration_seconds() -> None:
    assert format_duration(1.234) == "1.23 s"


# -------------------------------------------------------------------- structure


def test_refresh_builds_node_per_pipeline_entry(view: NodeGraphView) -> None:
    assert len(view._nodes) == 3
    assert [n.title for n in view._nodes] == ["Alpha", "Beta", "Gamma"]


def test_refresh_lays_out_horizontally(view: NodeGraphView) -> None:
    xs = [n.pos().x() for n in view._nodes]
    assert xs == [_layout_x(0), _layout_x(1), _layout_x(2)]
    # Spacing is exactly NODE_WIDTH + NODE_GAP between neighbours.
    assert xs[1] - xs[0] == NODE_WIDTH + NODE_GAP


def test_refresh_creates_one_edge_between_each_consecutive_pair(view: NodeGraphView) -> None:
    assert len(view._edges) == 2  # 3 nodes -> 2 edges


def test_empty_pipeline_has_no_nodes_and_no_edges(qapp: QApplication) -> None:
    empty = NodeGraphView(Pipeline())
    assert empty._nodes == []
    assert empty._edges == []


# -------------------------------------------------------------------- selection


def test_select_emits_signal_and_marks_node(view: NodeGraphView, qapp: QApplication) -> None:
    received: list[int] = []
    view.selection_changed.connect(received.append)
    view.select(1)
    qapp.processEvents()
    assert received == [1]
    assert view._nodes[1].selected is True
    assert view._nodes[0].selected is False


def test_clicking_node_body_emits_selection(view: NodeGraphView, qapp: QApplication) -> None:
    received: list[int] = []
    view.selection_changed.connect(received.append)
    view._nodes[2].body_clicked.emit(2)
    qapp.processEvents()
    assert received == [2]
    assert view._nodes[2].selected is True


# ---------------------------------------------------------------- enabled chip


def test_toggle_chip_flips_enabled_state(view: NodeGraphView, qapp: QApplication) -> None:
    fired: list[None] = []
    view.pipeline_changed.connect(lambda: fired.append(None))

    view._nodes[1].enabled = False  # simulate user clicking the chip
    view._nodes[1].enable_toggled.emit(1)
    qapp.processEvents()

    assert view._pipeline.nodes[1].enabled is False
    assert len(fired) == 1


# ---------------------------------------------------------------- remove chip


def test_remove_chip_drops_node_and_refreshes(view: NodeGraphView, qapp: QApplication) -> None:
    fired: list[None] = []
    view.pipeline_changed.connect(lambda: fired.append(None))

    view._nodes[1].remove_requested.emit(1)
    qapp.processEvents()

    assert len(view._pipeline.nodes) == 2
    assert [n.spec.name for n in view._pipeline.nodes] == ["Alpha", "Gamma"]
    assert len(view._nodes) == 2
    assert len(view._edges) == 1
    assert fired == [None]


# ---------------------------------------------------------------- drag reorder


def _drag_node(view: NodeGraphView, index: int, new_x: float) -> None:
    """Move the node to a new x-position and trigger the drag-release path."""
    view._nodes[index].setPos(QPointF(new_x, SCENE_MARGIN))
    view._nodes[index].drag_released.emit(index)


def test_drag_reorders_pipeline_by_x_position(view: NodeGraphView, qapp: QApplication) -> None:
    fired: list[None] = []
    view.pipeline_changed.connect(lambda: fired.append(None))

    # Move Alpha (index 0) past Gamma (index 2) by setting a very large x.
    _drag_node(view, 0, _layout_x(2) + 200)
    qapp.processEvents()

    assert [n.spec.name for n in view._pipeline.nodes] == ["Beta", "Gamma", "Alpha"]
    assert fired == [None]
    # After reorder, the visual nodes are re-laid-out from scratch.
    assert [n.title for n in view._nodes] == ["Beta", "Gamma", "Alpha"]


def test_drag_without_reorder_snaps_node_back_to_slot(view: NodeGraphView) -> None:
    original_x = view._nodes[1].pos().x()
    # Nudge Beta a little but not past its neighbours.
    view._nodes[1].setPos(QPointF(original_x + 10, SCENE_MARGIN))
    view._nodes[1].drag_released.emit(1)
    assert view._nodes[1].pos().x() == _layout_x(1)


def test_drag_remaps_timings(view: NodeGraphView, qapp: QApplication) -> None:
    view.set_timings([0.001, 0.002, 0.003])
    _drag_node(view, 0, _layout_x(2) + 200)
    qapp.processEvents()
    # Order is now [Beta, Gamma, Alpha]; timings should follow accordingly.
    assert view._nodes[0].timing == pytest.approx(0.002)
    assert view._nodes[1].timing == pytest.approx(0.003)
    assert view._nodes[2].timing == pytest.approx(0.001)


# --------------------------------------------------------------------- timings


def test_set_timings_updates_each_node(view: NodeGraphView) -> None:
    view.set_timings([0.001, None, 0.010])
    assert view._nodes[0].timing == pytest.approx(0.001)
    assert view._nodes[1].timing is None
    assert view._nodes[2].timing == pytest.approx(0.010)


def test_clear_timings_resets_all(view: NodeGraphView) -> None:
    view.set_timings([0.001, 0.002, 0.003])
    view.clear_timings()
    assert all(n.timing is None for n in view._nodes)


def test_refresh_preserves_last_known_timings(view: NodeGraphView) -> None:
    view.set_timings([0.001, 0.002, 0.003])
    view.refresh()
    assert view._nodes[0].timing == pytest.approx(0.001)
    assert view._nodes[2].timing == pytest.approx(0.003)


# ----------------------------------------------------------------------- ports


def test_default_node_shows_one_input_and_one_output_port(view: NodeGraphView) -> None:
    node = view._nodes[0]
    assert node.input_ports == ("in",)
    assert node.output_ports == ("out",)


def test_port_centres_sit_outside_the_node_body(view: NodeGraphView) -> None:
    node = view._nodes[0]
    inputs = node._port_centers(node.input_ports, side="left")
    outputs = node._port_centers(node.output_ports, side="right")
    assert inputs[0].x() < 0  # left of body
    assert outputs[0].x() > NODE_WIDTH  # right of body


def test_multi_input_node_stacks_ports_vertically(qapp: QApplication) -> None:
    multi_spec = OperationSpec(
        id="test.multi",
        name="Blend",
        category="Test",
        description="",
        parameters=(),
        func=lambda a, b: a,
        input_ports=("a", "b"),
    )
    pipeline = Pipeline()
    pipeline.add(multi_spec)
    widget = NodeGraphView(pipeline)
    node = widget._nodes[0]
    centres = node._port_centers(node.input_ports, side="left")
    assert len(centres) == 2
    assert centres[0].y() != centres[1].y()  # stacked, not overlapping


def _multi_input_spec(name: str = "Blend") -> OperationSpec:
    return OperationSpec(
        id=f"test.{name.lower()}",
        name=name,
        category="Test",
        description="",
        parameters=(),
        func=lambda a, b: a,
        input_ports=("a", "b"),
    )


def test_chain_edges_render_from_underlying_graph(qapp: QApplication) -> None:
    pipeline = Pipeline()
    pipeline.add(_spec("test.a", "Alpha"))
    pipeline.add(_spec("test.b", "Beta"))
    pipeline.add(_spec("test.c", "Gamma"))
    widget = NodeGraphView(pipeline)
    # Two chain edges (alpha→beta, beta→gamma); both render.
    assert len(widget._edges) == 2


def test_drag_to_connect_adds_edge_via_graph(qapp: QApplication) -> None:
    pipeline = Pipeline()
    alpha = pipeline.add(_spec("test.a", "Alpha"))
    pipeline.add(_spec("test.b", "Beta"))
    blend = pipeline.add(_multi_input_spec())
    widget = NodeGraphView(pipeline)
    # Sanity: blend's "a" port is chain-connected; "b" is not.
    connected_b = [
        e for e in pipeline.graph.edges if e.target == blend.id and e.target_port == "b"
    ]
    assert connected_b == []

    fired: list[None] = []
    widget.pipeline_changed.connect(lambda: fired.append(None))

    # Simulate dragging from Alpha's output to Blend's "b" input.
    src_item = widget._nodes_by_id[alpha.id]
    widget._begin_pending_edge(src_item, "out")
    target_scene = widget._nodes_by_id[blend.id].input_port_scene_position("b")
    widget._finalize_pending_edge(
        widget.mapFromScene(target_scene).toPointF()
        if hasattr(widget.mapFromScene(target_scene), "toPointF")
        else widget.mapFromScene(target_scene)
    )
    qapp.processEvents()

    edges_to_b = [
        e for e in pipeline.graph.edges if e.target == blend.id and e.target_port == "b"
    ]
    assert len(edges_to_b) == 1
    assert edges_to_b[0].source == alpha.id
    assert fired == [None]


def test_drag_to_disconnect_drops_edge_when_released_in_empty_space(
    qapp: QApplication,
) -> None:
    pipeline = Pipeline()
    pipeline.add(_spec("test.a", "Alpha"))
    pipeline.add(_spec("test.b", "Beta"))
    widget = NodeGraphView(pipeline)
    assert len(pipeline.graph.edges) == 1

    fired: list[None] = []
    widget.pipeline_changed.connect(lambda: fired.append(None))

    # Press on Beta's "in" port (which is chain-connected) → existing edge is
    # lifted. Then release in empty space → no new edge is created.
    widget._on_input_port_pressed(1, "in")
    widget._finalize_pending_edge(widget.mapFromScene(QPointF(10_000, 10_000)))
    qapp.processEvents()
    assert pipeline.graph.edges == []
    # Each step that mutates the graph emits pipeline_changed.
    assert fired


def test_drag_to_connect_rejects_cycle(qapp: QApplication) -> None:
    pipeline = Pipeline()
    pipeline.add(_spec("test.a", "Alpha"))
    pipeline.add(_spec("test.b", "Beta"))
    blend = pipeline.add(_multi_input_spec())
    widget = NodeGraphView(pipeline)
    edges_before = len(pipeline.graph.edges)
    # Try to wire blend's output back into alpha — would form a cycle.
    src_item = widget._nodes_by_id[blend.id]
    widget._begin_pending_edge(src_item, "out")
    alpha_in_scene = widget._nodes_by_id[pipeline.nodes[0].id].input_port_scene_position("in")
    widget._finalize_pending_edge(widget.mapFromScene(alpha_in_scene))
    qapp.processEvents()
    assert len(pipeline.graph.edges) == edges_before


def test_input_port_scene_position_looks_up_by_name(view: NodeGraphView) -> None:
    node = view._nodes[0]
    point = node.input_port_scene_position("in")
    expected = node.mapToScene(node._port_centers(node.input_ports, side="left")[0])
    assert point == expected
    with pytest.raises(KeyError):
        node.input_port_scene_position("bogus")
