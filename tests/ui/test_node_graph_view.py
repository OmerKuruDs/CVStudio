from __future__ import annotations

import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtWidgets import QApplication

from cvstudio.core.operation import OperationSpec, Parameter
from cvstudio.core.pipeline import Pipeline
from cvstudio.ui.node_graph_view import (
    NODE_GAP,
    NODE_WIDTH,
    SCENE_MARGIN,
    NodeGraphView,
    NodeItem,
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


def _chain_items(widget: NodeGraphView) -> list[NodeItem]:
    """Return the NodeItems that correspond to user-added chain ops (Source
    excluded). Tests written before Phase 2c used this view in place of
    `widget._nodes`."""
    return [item for item in widget._nodes if not item.is_source]


# --------------------------------------------------------------- format_duration


def test_format_duration_microseconds() -> None:
    assert format_duration(0.000_25) == "250 µs"


def test_format_duration_milliseconds() -> None:
    assert format_duration(0.012_3) == "12.3 ms"


def test_format_duration_seconds() -> None:
    assert format_duration(1.234) == "1.23 s"


# -------------------------------------------------------------------- structure


def test_refresh_builds_node_per_pipeline_entry(view: NodeGraphView) -> None:
    chain_items = _chain_items(view)
    assert len(chain_items) == 3
    assert [n.title for n in chain_items] == ["Alpha", "Beta", "Gamma"]


def test_refresh_lays_out_horizontally(view: NodeGraphView) -> None:
    # Source occupies graph index 0; chain ops start at 1.
    chain_items = _chain_items(view)
    xs = [n.pos().x() for n in chain_items]
    assert xs == [_layout_x(1), _layout_x(2), _layout_x(3)]
    assert xs[1] - xs[0] == NODE_WIDTH + NODE_GAP


def test_refresh_creates_one_edge_between_each_consecutive_pair(view: NodeGraphView) -> None:
    # Source -> Alpha + Alpha -> Beta + Beta -> Gamma = 3 edges.
    assert len(view._edges) == 3


def test_empty_pipeline_renders_only_the_source_node(qapp: QApplication) -> None:
    empty = NodeGraphView(Pipeline())
    assert len(empty._nodes) == 1
    assert empty._nodes[0].is_source
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
    # Gamma is at chain index 2 / graph index 3 (Source is at graph index 0).
    gamma_item = view._nodes_by_id[view._pipeline.nodes[2].id]
    gamma_item.body_clicked.emit(gamma_item.index)
    qapp.processEvents()
    # Selection signal emits chain index for ParameterPanel compatibility.
    assert received == [2]
    assert gamma_item.selected is True


# ---------------------------------------------------------------- enabled chip


def test_toggle_chip_flips_enabled_state(view: NodeGraphView, qapp: QApplication) -> None:
    fired: list[None] = []
    view.pipeline_changed.connect(lambda: fired.append(None))

    beta_item = view._nodes_by_id[view._pipeline.nodes[1].id]
    beta_item.enabled = False  # simulate user clicking the chip
    beta_item.enable_toggled.emit(beta_item.index)
    qapp.processEvents()

    assert view._pipeline.nodes[1].enabled is False
    assert len(fired) == 1


# ---------------------------------------------------------------- remove chip


def test_remove_chip_drops_node_and_refreshes(view: NodeGraphView, qapp: QApplication) -> None:
    fired: list[None] = []
    view.pipeline_changed.connect(lambda: fired.append(None))

    beta_item = view._nodes_by_id[view._pipeline.nodes[1].id]
    beta_item.remove_requested.emit(beta_item.index)
    qapp.processEvents()

    assert len(view._pipeline.nodes) == 2
    assert [n.spec.name for n in view._pipeline.nodes] == ["Alpha", "Gamma"]
    # Source + remaining 2 chain nodes = 3 graph nodes; 2 chain edges.
    assert len(view._nodes) == 3
    assert len(view._edges) == 2
    assert fired == [None]


# ---------------------------------------------------------------- drag reorder


def _drag_chain_node(view: NodeGraphView, chain_index: int, new_x: float, new_y: float = SCENE_MARGIN) -> None:
    """Move the chain op at `chain_index` to a new (x, y) and fire the
    drag-release signal. Source lives at graph index 0, so chain index `i`
    sits at graph index `i + 1`."""
    pipeline_node = view._pipeline.nodes[chain_index]
    item = view._nodes_by_id[pipeline_node.id]
    item.setPos(QPointF(new_x, new_y))
    item.drag_released.emit(item.index)


def test_drag_persists_node_position_on_graph_node(view: NodeGraphView) -> None:
    _drag_chain_node(view, 1, new_x=425.0, new_y=180.0)  # move Beta freely
    beta = view._pipeline.nodes[1]
    assert beta.position == (425.0, 180.0)


def test_drag_does_not_reorder_chain(view: NodeGraphView, qapp: QApplication) -> None:
    """Phase 2d drops drag-to-reorder. Position changes are purely visual; the
    chain stays in add-order."""
    fired: list[None] = []
    view.pipeline_changed.connect(lambda: fired.append(None))

    _drag_chain_node(view, 0, _layout_x(99))  # Alpha dragged far to the right
    qapp.processEvents()

    assert [n.spec.name for n in view._pipeline.nodes] == ["Alpha", "Beta", "Gamma"]
    # No pipeline_changed signal — only Position changed, not topology.
    assert fired == []


def test_dragging_source_node_persists_its_position(view: NodeGraphView) -> None:
    """The Source node is freely movable just like a regular chain node — its
    position rides through GraphNode.position and serialization."""
    source_item = view._nodes[0]
    assert source_item.is_source
    source_item.setPos(QPointF(40.0, 220.0))
    source_item.drag_released.emit(source_item.index)
    source_node = view._pipeline.graph.get_node(view._pipeline.source_node_id)
    assert source_node.position == (40.0, 220.0)


def test_refresh_honours_persisted_position(view: NodeGraphView) -> None:
    _drag_chain_node(view, 0, new_x=900.0, new_y=120.0)  # Alpha somewhere new
    view.refresh()
    alpha_item = view._nodes_by_id[view._pipeline.nodes[0].id]
    assert alpha_item.pos().x() == pytest.approx(900.0)
    assert alpha_item.pos().y() == pytest.approx(120.0)


# --------------------------------------------------------------------- timings


def test_set_timings_updates_each_node(view: NodeGraphView) -> None:
    view.set_timings([0.001, None, 0.010])
    chain_items = _chain_items(view)
    assert chain_items[0].timing == pytest.approx(0.001)
    assert chain_items[1].timing is None
    assert chain_items[2].timing == pytest.approx(0.010)


def test_clear_timings_resets_all(view: NodeGraphView) -> None:
    view.set_timings([0.001, 0.002, 0.003])
    view.clear_timings()
    assert all(n.timing is None for n in _chain_items(view))


def test_refresh_preserves_last_known_timings(view: NodeGraphView) -> None:
    view.set_timings([0.001, 0.002, 0.003])
    view.refresh()
    chain_items = _chain_items(view)
    assert chain_items[0].timing == pytest.approx(0.001)
    assert chain_items[2].timing == pytest.approx(0.003)


# ----------------------------------------------------------------------- ports


def test_default_node_shows_one_input_and_one_output_port(view: NodeGraphView) -> None:
    node = _chain_items(view)[0]  # Alpha (first chain op, not Source)
    assert node.input_ports == ("in",)
    assert node.output_ports == ("out",)


def test_port_centres_sit_outside_the_node_body(view: NodeGraphView) -> None:
    node = _chain_items(view)[0]
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
    node = _chain_items(widget)[0]
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
    # Source→Alpha, Alpha→Beta, Beta→Gamma = 3 edges (Source is auto-chained too).
    assert len(widget._edges) == 3


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
    # Source → Alpha and Alpha → Beta auto-edges.
    assert len(pipeline.graph.edges) == 2

    fired: list[None] = []
    widget.pipeline_changed.connect(lambda: fired.append(None))

    # Find Beta's NodeItem index in graph order so the disconnect handler
    # operates on the right node.
    beta_id = pipeline.nodes[1].id
    beta_graph_index = next(
        i for i, gn in enumerate(pipeline.graph.nodes) if gn.id == beta_id
    )
    widget._on_input_port_pressed(beta_graph_index, "in")
    widget._finalize_pending_edge(widget.mapFromScene(QPointF(10_000, 10_000)))
    qapp.processEvents()
    # Alpha → Beta edge was lifted and dropped in empty space. Source → Alpha
    # remains — it is a chain-managed edge unrelated to the user's action.
    edges = pipeline.graph.edges
    assert len(edges) == 1
    assert edges[0].target == pipeline.nodes[0].id  # Alpha's incoming edge from Source
    assert fired


def test_clicking_input_port_via_real_mouse_event_fires_signal(
    qapp: QApplication,
) -> None:
    """Regression: NodeItem.boundingRect must cover the port circles, or Qt
    silently drops mouse presses landing on them. Without this, drag-to-connect
    is dead because the underlying signal never fires."""
    from PySide6.QtCore import QEvent
    from PySide6.QtWidgets import QGraphicsSceneMouseEvent

    pipeline = Pipeline()
    pipeline.add(_spec("test.a", "Alpha"))
    pipeline.add(_multi_input_spec())
    widget = NodeGraphView(pipeline)
    widget.show()
    qapp.processEvents()

    blend_item = widget._nodes_by_id[pipeline.nodes[1].id]
    port_scene = blend_item.input_port_scene_position("b")
    found = widget._scene.itemAt(port_scene, widget.transform())
    assert isinstance(found, NodeItem)
    assert found is blend_item

    received: list[tuple[int, str]] = []
    blend_item.input_port_pressed.connect(lambda i, n: received.append((i, n)))
    local = blend_item.mapFromScene(port_scene)
    event = QGraphicsSceneMouseEvent(QEvent.Type.GraphicsSceneMousePress)
    event.setPos(local)
    event.setButton(Qt.MouseButton.LeftButton)
    blend_item.mousePressEvent(event)
    qapp.processEvents()
    # Blend sits at graph index 2 (Source=0, Alpha=1, Blend=2).
    assert received == [(blend_item.index, "b")]


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
    node = _chain_items(view)[0]
    point = node.input_port_scene_position("in")
    expected = node.mapToScene(node._port_centers(node.input_ports, side="left")[0])
    assert point == expected
    with pytest.raises(KeyError):
        node.input_port_scene_position("bogus")
