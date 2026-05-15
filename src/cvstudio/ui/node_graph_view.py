"""NodeGraphView — visual representation of the pipeline as a horizontal
node chain.

Drop-in replacement for the QListWidget-based PipelineView. Each node is a
rounded rectangle painted on a QGraphicsScene; edges are bezier curves
connecting consecutive (enabled or disabled) nodes. Behaviours preserved
from the old view:

    * click a node     — emits `selection_changed(index)`
    * drag a node      — releases reorder the pipeline by x-position
    * toggle dot       — enables / disables that node (top-left chip)
    * X chip           — removes the node (top-right chip)
    * `set_timings`    — annotates each node with its last execution time

The underlying `Pipeline` data model is unchanged — this is a presentation
layer. A future iteration will replace the linear `Pipeline` with a true DAG
and reuse this same view with port-aware edges.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
)
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsObject,
    QGraphicsPathItem,
    QGraphicsScene,
    QGraphicsSceneMouseEvent,
    QGraphicsView,
    QStyleOptionGraphicsItem,
    QWidget,
)

from cvstudio.core.graph import GraphEdge
from cvstudio.core.pipeline import Pipeline

NODE_WIDTH = 150
NODE_HEIGHT = 64
NODE_GAP = 56
SCENE_MARGIN = 24

CHIP_RADIUS = 7
CHIP_INSET = 8

PORT_RADIUS = 6
PORT_GAP = 14  # vertical gap between adjacent ports on the same edge
PORT_HIT_PAD = 3  # extra forgiveness around each port for mouse hit-testing
PORT_BOUND_PAD = 2 * PORT_RADIUS + PORT_HIT_PAD + 2  # space the boundingRect needs to grant on either side

EDGE_COLOR = QColor("#0891b2")
EDGE_WIDTH = 2
PORT_FILL = QColor("#ffffff")
PORT_BORDER = QColor("#0e7490")

DISABLED_ALPHA = 110

_CATEGORY_COLORS: dict[str, QColor] = {
    "Filtering": QColor("#67e8f9"),   # bright cyan
    "Threshold": QColor("#86efac"),   # pastel mint
    "Morphology": QColor("#c4b5fd"),  # pastel violet
    "Edge": QColor("#fdba74"),         # pastel orange
    "Color": QColor("#f9a8d4"),        # pastel pink
    "Geometric": QColor("#7dd3fc"),    # sky
    "Analysis": QColor("#fde68a"),     # pastel yellow
    "Composite": QColor("#a78bfa"),    # vivid violet
    "Source": QColor("#94a3b8"),       # slate
}
_DEFAULT_CATEGORY_COLOR = QColor("#cbd5e1")


def format_duration(seconds: float) -> str:
    """Format a duration in seconds as a short human-readable string."""
    if seconds < 1e-3:
        return f"{seconds * 1e6:.0f} µs"
    if seconds < 1.0:
        return f"{seconds * 1e3:.1f} ms"
    return f"{seconds:.2f} s"


def _category_color(category: str) -> QColor:
    return _CATEGORY_COLORS.get(category, _DEFAULT_CATEGORY_COLOR)


def _layout_x(index: int) -> float:
    return SCENE_MARGIN + index * (NODE_WIDTH + NODE_GAP)


class NodeItem(QGraphicsObject):
    """One pipeline node, rendered as a rounded rectangle with two small
    interactive chips (toggle and remove)."""

    body_clicked = Signal(int)
    enable_toggled = Signal(int)
    remove_requested = Signal(int)
    moved = Signal()
    drag_released = Signal(int)
    output_port_pressed = Signal(int, str)
    input_port_pressed = Signal(int, str)

    def __init__(
        self,
        index: int,
        node_id: str,
        title: str,
        category: str,
        enabled: bool,
        input_ports: tuple[str, ...] = ("in",),
        output_ports: tuple[str, ...] = ("out",),
        is_source: bool = False,
        parent: QGraphicsItem | None = None,
    ) -> None:
        super().__init__(parent)
        self.index = index
        self.node_id = node_id
        self.title = title
        self.category = category
        self.enabled = enabled
        self.timing: float | None = None
        self.selected = False
        self.input_ports = input_ports
        self.output_ports = output_ports
        self.is_source = is_source
        self._color = _category_color(category)
        self._dragged = False
        # Source is freely movable like any other node — the user can park it
        # wherever the wires look tidiest. Chips are still suppressed so it
        # remains non-removable and always enabled.
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)

    # ------------------------------------------------------------------ painting

    def boundingRect(self) -> QRectF:  # noqa: N802 (Qt override)
        # Extend the rect so the port circles on either edge sit inside it;
        # Qt only routes mouse events to items whose boundingRect contains the
        # cursor, so ports drawn outside this region would otherwise be unhit-
        # testable.
        return QRectF(
            -PORT_BOUND_PAD,
            0,
            NODE_WIDTH + 2 * PORT_BOUND_PAD,
            NODE_HEIGHT,
        )

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: QWidget | None = None,
    ) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        # The body fills the node's NODE_WIDTH by NODE_HEIGHT box; the
        # boundingRect is wider to absorb port hit-tests but visual painting
        # of the rounded body should stay on the original geometry.
        rect = QRectF(0.5, 0.5, NODE_WIDTH - 1, NODE_HEIGHT - 1)

        body_color = QColor(self._color)
        if not self.enabled:
            body_color.setAlpha(DISABLED_ALPHA)
        painter.setBrush(QBrush(body_color))

        border_color = QColor("#0891b2") if self.selected else QColor("#94a3b8")
        border_width = 3 if self.selected else 1
        painter.setPen(QPen(border_color, border_width))
        painter.drawRoundedRect(rect, 10, 10)

        # Category strip on the left edge.
        strip = QRectF(rect.left(), rect.top(), 6, rect.height())
        accent = QColor(self._color).darker(140)
        painter.setBrush(accent)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(strip, 3, 3)

        # Title — dark navy reads well on the pastel category fills.
        painter.setPen(QColor("#0b1437"))
        title_font = QFont(painter.font())
        title_font.setBold(True)
        painter.setFont(title_font)
        title_rect = rect.adjusted(14, 6, -10, -22)
        painter.drawText(
            title_rect,
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            self.title,
        )

        # Subtitle: category + timing.
        info_font = QFont(painter.font())
        info_font.setBold(False)
        info_font.setPointSize(max(7, info_font.pointSize() - 2))
        painter.setFont(info_font)
        painter.setPen(QColor("#475569"))
        info_text = self.category
        if self.timing is not None:
            info_text = f"{self.category}  ·  {format_duration(self.timing)}"
        info_rect = rect.adjusted(14, rect.height() - 22, -10, -4)
        painter.drawText(
            info_rect,
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            info_text,
        )

        # Input ports on the left edge, output ports on the right edge.
        painter.setBrush(QBrush(PORT_FILL))
        painter.setPen(QPen(PORT_BORDER, 1.5))
        for center in self._port_centers(self.input_ports, side="left"):
            painter.drawEllipse(center, PORT_RADIUS, PORT_RADIUS)
        for center in self._port_centers(self.output_ports, side="right"):
            painter.drawEllipse(center, PORT_RADIUS, PORT_RADIUS)

        # Source nodes are immutable controls — skip the toggle/remove chips.
        if not self.is_source:
            # Toggle chip (top-left within the body).
            toggle_center = self._toggle_chip_center()
            painter.setPen(QPen(QColor("#1d2129"), 1))
            painter.setBrush(QColor("#cef1d2") if self.enabled else QColor("#3a3f48"))
            painter.drawEllipse(toggle_center, CHIP_RADIUS, CHIP_RADIUS)

            # Remove chip (top-right).
            remove_center = self._remove_chip_center()
            painter.setBrush(QColor("#3a3f48"))
            painter.drawEllipse(remove_center, CHIP_RADIUS, CHIP_RADIUS)
            painter.setPen(QPen(QColor("#ffffff"), 1.5))
            offset = CHIP_RADIUS - 3
            painter.drawLine(
                QPointF(remove_center.x() - offset, remove_center.y() - offset),
                QPointF(remove_center.x() + offset, remove_center.y() + offset),
            )
            painter.drawLine(
                QPointF(remove_center.x() - offset, remove_center.y() + offset),
                QPointF(remove_center.x() + offset, remove_center.y() - offset),
            )

    # ------------------------------------------------------------------ hit-tests

    def _toggle_chip_center(self) -> QPointF:
        return QPointF(NODE_WIDTH - 2 * (CHIP_INSET + CHIP_RADIUS) - 4, CHIP_INSET + CHIP_RADIUS - 2)

    def _remove_chip_center(self) -> QPointF:
        return QPointF(NODE_WIDTH - CHIP_INSET - CHIP_RADIUS, CHIP_INSET + CHIP_RADIUS - 2)

    def _hit_chip(self, point: QPointF, center: QPointF) -> bool:
        dx = point.x() - center.x()
        dy = point.y() - center.y()
        return dx * dx + dy * dy <= (CHIP_RADIUS + 2) ** 2

    def _port_centers(self, ports: tuple[str, ...], *, side: str) -> list[QPointF]:
        """Compute the (x, y) of each port circle in node-local coords."""
        if not ports:
            return []
        x = -PORT_RADIUS if side == "left" else NODE_WIDTH + PORT_RADIUS
        n = len(ports)
        total = (n - 1) * PORT_GAP
        first_y = NODE_HEIGHT / 2 - total / 2
        return [QPointF(x, first_y + i * PORT_GAP) for i in range(n)]

    def input_port_scene_position(self, port_name: str) -> QPointF:
        """Scene-coords centre of the named input port. Useful for edge drawing."""
        for name, center in zip(self.input_ports, self._port_centers(self.input_ports, side="left"), strict=True):
            if name == port_name:
                return self.mapToScene(center)
        raise KeyError(port_name)

    def output_port_scene_position(self, port_name: str) -> QPointF:
        for name, center in zip(self.output_ports, self._port_centers(self.output_ports, side="right"), strict=True):
            if name == port_name:
                return self.mapToScene(center)
        raise KeyError(port_name)

    def _port_at_local(self, point: QPointF) -> tuple[str, str] | None:
        """If `point` (in node-local coords) lands on a port circle, return
        ('input' | 'output', port_name); else None."""
        for name, center in zip(
            self.input_ports,
            self._port_centers(self.input_ports, side="left"),
            strict=True,
        ):
            if self._hit_port(point, center):
                return ("input", name)
        for name, center in zip(
            self.output_ports,
            self._port_centers(self.output_ports, side="right"),
            strict=True,
        ):
            if self._hit_port(point, center):
                return ("output", name)
        return None

    @staticmethod
    def _hit_port(point: QPointF, center: QPointF) -> bool:
        dx = point.x() - center.x()
        dy = point.y() - center.y()
        return dx * dx + dy * dy <= (PORT_RADIUS + PORT_HIT_PAD) ** 2

    # ------------------------------------------------------------------ events

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        pos = event.pos()
        port = self._port_at_local(pos)
        if port is not None:
            side, port_name = port
            if side == "output":
                self.output_port_pressed.emit(self.index, port_name)
            else:
                self.input_port_pressed.emit(self.index, port_name)
            event.accept()  # never start a node move when grabbing a port
            return
        # Chip interactions don't apply to the Source node.
        if not self.is_source:
            if self._hit_chip(pos, self._toggle_chip_center()):
                self.enabled = not self.enabled
                self.update()
                self.enable_toggled.emit(self.index)
                event.accept()
                return
            if self._hit_chip(pos, self._remove_chip_center()):
                self.remove_requested.emit(self.index)
                event.accept()
                return
        self._dragged = False
        self.body_clicked.emit(self.index)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:  # noqa: N802
        self._dragged = True
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:  # noqa: N802
        super().mouseReleaseEvent(event)
        if event.button() == Qt.MouseButton.LeftButton and self._dragged:
            self.drag_released.emit(self.index)
        self._dragged = False

    def itemChange(  # noqa: N802 (Qt override)
        self, change: QGraphicsItem.GraphicsItemChange, value: Any
    ) -> Any:
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self.moved.emit()
        return super().itemChange(change, value)


class EdgeItem(QGraphicsPathItem):
    """Bezier connector between two NodeItems' typed ports."""

    def __init__(
        self,
        source: NodeItem,
        target: NodeItem,
        source_port: str = "out",
        target_port: str = "in",
    ) -> None:
        super().__init__()
        self.source = source
        self.target = target
        self.source_port = source_port
        self.target_port = target_port
        self.setPen(QPen(EDGE_COLOR, EDGE_WIDTH))
        self.setZValue(-1)  # draw under nodes
        source.moved.connect(self.refresh)
        target.moved.connect(self.refresh)
        self.refresh()

    def refresh(self) -> None:
        sp = self.source.output_port_scene_position(self.source_port)
        tp = self.target.input_port_scene_position(self.target_port)
        dx = max(40, (tp.x() - sp.x()) / 2)
        c1 = QPointF(sp.x() + dx, sp.y())
        c2 = QPointF(tp.x() - dx, tp.y())
        path = QPainterPath(sp)
        path.cubicTo(c1, c2, tp)
        self.setPath(path)


class NodeGraphView(QGraphicsView):
    selection_changed = Signal(int)
    pipeline_changed = Signal()

    def __init__(self, pipeline: Pipeline, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pipeline = pipeline
        self._timings: list[float | None] = []
        self._nodes: list[NodeItem] = []
        self._nodes_by_id: dict[str, NodeItem] = {}
        self._edges: list[EdgeItem] = []
        self._selected_index = -1
        self._pending_source: tuple[NodeItem, str] | None = None
        self._pending_wire: QGraphicsPathItem | None = None

        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        # Pale cool background ties the graph into the global pastel theme.
        self.setBackgroundBrush(QColor("#f1f5f9"))
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setMinimumHeight(NODE_HEIGHT + 2 * SCENE_MARGIN + 16)

        self.refresh()

    # ------------------------------------------------------------------ public API

    def refresh(self) -> None:
        # Preserve current timings but trim them to the new length.
        del self._timings[len(self._pipeline.nodes):]
        previous_selected = self._selected_index
        self._scene.clear()
        self._nodes.clear()
        self._nodes_by_id.clear()
        self._edges.clear()
        self._pending_wire = None  # was a child of scene — already gone

        source_id = self._pipeline.source_node_id

        # Render in graph order: Source first, then chain ops. Source occupies
        # column 0 of the auto-layout so chain ops keep their familiar positions.
        ordered_graph_nodes = [
            self._pipeline.graph.get_node(source_id),
            *self._pipeline.nodes,
        ]
        for i, node in enumerate(ordered_graph_nodes):
            is_source = node.id == source_id
            chain_index = self._pipeline.chain_index_of(node.id)
            item = NodeItem(
                index=i,
                node_id=node.id,
                title=node.spec.name,
                category=node.spec.category,
                enabled=node.enabled,
                input_ports=node.spec.input_ports,
                output_ports=node.spec.output_ports,
                is_source=is_source,
            )
            # Honour a persisted position from a previous drag (Source included);
            # otherwise auto-layout. Source defaults to column 0, chain nodes
            # cascade rightwards from column 1.
            if node.position is not None:
                item.setPos(node.position[0], node.position[1])
            elif is_source:
                item.setPos(_layout_x(0), SCENE_MARGIN)
            else:
                item.setPos(_layout_x(i), SCENE_MARGIN)
            # Timings are indexed by chain position; Source has no timing.
            if not is_source and 0 <= chain_index < len(self._timings):
                item.timing = self._timings[chain_index]
            self._scene.addItem(item)
            self._nodes.append(item)
            self._nodes_by_id[node.id] = item

            item.body_clicked.connect(self._on_node_clicked)
            item.enable_toggled.connect(self._on_node_toggled)
            item.remove_requested.connect(self._on_node_remove_requested)
            item.drag_released.connect(self._on_node_drag_released)
            item.output_port_pressed.connect(self._on_output_port_pressed)
            item.input_port_pressed.connect(self._on_input_port_pressed)

        # Render every edge that the underlying Graph currently holds — chain
        # edges plus any user-drawn multi-input wires.
        for graph_edge in self._pipeline.graph.edges:
            src_item = self._nodes_by_id.get(graph_edge.source)
            tgt_item = self._nodes_by_id.get(graph_edge.target)
            if src_item is None or tgt_item is None:
                continue
            edge = EdgeItem(
                src_item,
                tgt_item,
                source_port=graph_edge.source_port,
                target_port=graph_edge.target_port,
            )
            self._scene.addItem(edge)
            self._edges.append(edge)

        if self._nodes:
            right = _layout_x(len(self._nodes) - 1) + NODE_WIDTH + SCENE_MARGIN
            self._scene.setSceneRect(0, 0, right, NODE_HEIGHT + 2 * SCENE_MARGIN)
        else:
            self._scene.setSceneRect(0, 0, 200, NODE_HEIGHT + 2 * SCENE_MARGIN)

        new_selection = (
            previous_selected if 0 <= previous_selected < len(self._nodes) else -1
        )
        self._apply_selection(new_selection)

    def select(self, index: int) -> None:
        if not (0 <= index < len(self._nodes)):
            index = -1
        self._apply_selection(index)
        self.selection_changed.emit(index)

    def set_timings(self, timings: Sequence[float | None]) -> None:
        """Update per-chain-node timing. Source is excluded — it has no
        execution step. Iteration walks the chain in order."""
        self._timings = list(timings)
        chain_index = 0
        for node_item in self._nodes:
            if node_item.is_source:
                node_item.timing = None
                continue
            node_item.timing = (
                self._timings[chain_index]
                if chain_index < len(self._timings)
                else None
            )
            node_item.update()
            chain_index += 1

    def clear_timings(self) -> None:
        self.set_timings([None] * len(self._pipeline.nodes))

    # ------------------------------------------------------------------ internals

    def _apply_selection(self, index: int) -> None:
        self._selected_index = index
        for i, item in enumerate(self._nodes):
            item.selected = i == index
            item.update()

    def _on_node_clicked(self, index: int) -> None:
        self._apply_selection(index)
        # Selection signal carries chain index for ParameterPanel compatibility;
        # clicking the Source node clears the param panel (chain index = -1).
        node_id = self._nodes[index].node_id
        self.selection_changed.emit(self._pipeline.chain_index_of(node_id))

    def _on_node_toggled(self, index: int) -> None:
        if not (0 <= index < len(self._nodes)):
            return
        node_id = self._nodes[index].node_id
        self._pipeline.graph.get_node(node_id).enabled = self._nodes[index].enabled
        self.pipeline_changed.emit()

    def _on_node_remove_requested(self, index: int) -> None:
        if not (0 <= index < len(self._nodes)):
            return
        node_id = self._nodes[index].node_id
        chain_index = self._pipeline.chain_index_of(node_id)
        if chain_index < 0:
            return  # Source node — not removable
        self._pipeline.remove(chain_index)
        if chain_index < len(self._timings):
            del self._timings[chain_index]
        if self._selected_index == index:
            self._selected_index = -1
        elif self._selected_index > index:
            self._selected_index -= 1
        self.refresh()
        sel_node = (
            self._nodes[self._selected_index]
            if 0 <= self._selected_index < len(self._nodes)
            else None
        )
        self.selection_changed.emit(
            self._pipeline.chain_index_of(sel_node.node_id) if sel_node else -1
        )
        self.pipeline_changed.emit()

    def _on_node_drag_released(self, index: int) -> None:
        """Persist the dropped position on the GraphNode (including the Source
        node) so it survives refreshes and serialization round-trips."""
        if not (0 <= index < len(self._nodes)):
            return
        item = self._nodes[index]
        pos = item.pos()
        node = self._pipeline.graph.get_node(item.node_id)
        node.position = (pos.x(), pos.y())

    # ---------------------------------------------------------- drag-to-connect

    def _on_output_port_pressed(self, index: int, port_name: str) -> None:
        if not (0 <= index < len(self._nodes)):
            return
        self._begin_pending_edge(self._nodes[index], port_name)

    def _on_input_port_pressed(self, index: int, port_name: str) -> None:
        """Pressing on a connected input port lifts that edge — the pending
        connection becomes anchored at the original source so the user can
        re-target or release into empty space to detach."""
        if not (0 <= index < len(self._nodes)):
            return
        target_id = self._nodes[index].node_id
        existing = next(
            (
                e
                for e in self._pipeline.graph.edges
                if e.target == target_id and e.target_port == port_name
            ),
            None,
        )
        if existing is None:
            return
        self._pipeline.graph.remove_edge(existing)
        self.refresh()
        self.pipeline_changed.emit()
        # After refresh, the visible NodeItems are fresh instances — look up
        # the source by node id rather than holding a stale reference.
        new_source = self._nodes_by_id.get(existing.source)
        if new_source is not None:
            self._begin_pending_edge(new_source, existing.source_port)

    def _begin_pending_edge(self, source: NodeItem, source_port: str) -> None:
        self._pending_source = (source, source_port)
        pen = QPen(EDGE_COLOR, EDGE_WIDTH, Qt.PenStyle.DashLine)
        pen.setCosmetic(True)
        wire = QGraphicsPathItem()
        wire.setPen(pen)
        wire.setZValue(10)
        self._scene.addItem(wire)
        self._pending_wire = wire
        # Seed at the source port; mouseMoveEvent will keep it in sync.
        start = source.output_port_scene_position(source_port)
        self._update_pending_wire(start)

    def _update_pending_wire(self, end_scene: QPointF) -> None:
        if self._pending_source is None or self._pending_wire is None:
            return
        src, port = self._pending_source
        sp = src.output_port_scene_position(port)
        dx = max(40, (end_scene.x() - sp.x()) / 2)
        c1 = QPointF(sp.x() + dx, sp.y())
        c2 = QPointF(end_scene.x() - dx, end_scene.y())
        path = QPainterPath(sp)
        path.cubicTo(c1, c2, end_scene)
        self._pending_wire.setPath(path)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 (Qt override)
        if self._pending_source is not None:
            scene_pos = self.mapToScene(event.position().toPoint())
            self._update_pending_wire(scene_pos)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802 (Qt override)
        if (
            self._pending_source is not None
            and event.button() == Qt.MouseButton.LeftButton
        ):
            self._finalize_pending_edge(event.position())
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _finalize_pending_edge(self, viewport_pos: QPointF) -> None:
        assert self._pending_source is not None
        source_item, source_port = self._pending_source
        self._pending_source = None
        if self._pending_wire is not None:
            self._scene.removeItem(self._pending_wire)
            self._pending_wire = None

        # `viewport_pos` may be a QPoint or QPointF depending on the caller
        # (real Qt events use QPointF; tests pass QPoint). Normalize.
        point = (
            viewport_pos.toPoint() if hasattr(viewport_pos, "toPoint") else viewport_pos
        )
        if not isinstance(point, QPoint):
            point = QPoint(int(point.x()), int(point.y()))
        scene_pos = self.mapToScene(point)
        target = self._port_under_scene_pos(scene_pos)
        if target is None:
            return  # released over empty space → leave the source disconnected
        target_item, target_port = target
        if target_item is source_item:
            return  # cannot wire a node back into itself

        try:
            self._pipeline.graph.add_edge(
                GraphEdge(
                    source=source_item.node_id,
                    source_port=source_port,
                    target=target_item.node_id,
                    target_port=target_port,
                )
            )
        except ValueError:
            return  # cycle / duplicate / unknown port — silently reject
        self.refresh()
        self.pipeline_changed.emit()

    def _port_under_scene_pos(
        self, scene_pos: QPointF
    ) -> tuple[NodeItem, str] | None:
        """Find the input port (if any) at `scene_pos`. Walks the parent chain
        of the hit item until a NodeItem is found, then asks it whether the
        local position falls on a port circle."""
        item = self._scene.itemAt(scene_pos, self.transform())
        while item is not None and not isinstance(item, NodeItem):
            item = item.parentItem()
        if item is None:
            # Fallback: search all node items by hit-testing each port.
            for node_item in self._nodes:
                local = node_item.mapFromScene(scene_pos)
                port = node_item._port_at_local(local)
                if port is not None and port[0] == "input":
                    return node_item, port[1]
            return None
        local = item.mapFromScene(scene_pos)
        port = item._port_at_local(local)
        if port is None or port[0] != "input":
            return None
        return item, port[1]
