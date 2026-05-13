"""Graph — DAG-aware data model that powers Pipeline under the hood.

A `Graph` holds a dictionary of `GraphNode` instances (one per pipeline step)
and a list of `GraphEdge` instances connecting their typed input/output
ports. Execution is a topological sort followed by per-node evaluation: for
each node, inputs are gathered from incoming edges (or fall back to the
graph's input image if a port has no incoming connection), the node's spec
function runs, and outputs are cached by port for downstream consumers.

For now `Pipeline` keeps its linear chain abstraction on top of this; the
graph is degenerate (single linear path) until the visual editor exposes
port-level wiring. The data model itself already supports branching, merging,
multi-input ops, and dead-branch culling — they just have no UI yet.
"""

from __future__ import annotations

import contextlib
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cvsandbox.core.operation import OperationSpec

NodeId = str


def _auto_node_id() -> NodeId:
    return f"_n{uuid.uuid4().hex[:8]}"


@dataclass(frozen=True, slots=True)
class GraphEdge:
    source: NodeId
    target: NodeId
    source_port: str = "out"
    target_port: str = "in"


@dataclass
class GraphNode:
    spec: OperationSpec
    id: NodeId = field(default_factory=_auto_node_id)
    params: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    position: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        defaults = self.spec.default_params()
        unknown = set(self.params) - set(defaults)
        if unknown:
            raise ValueError(f"Unknown parameter(s) for {self.spec.id}: {sorted(unknown)}")
        for name, default in defaults.items():
            self.params.setdefault(name, default)

    def call(self, *inputs: np.ndarray) -> np.ndarray | tuple[np.ndarray, ...]:
        return self.spec.func(*inputs, **self.params)


class Graph:
    """A directed acyclic graph of operation nodes with typed ports."""

    def __init__(self) -> None:
        self._nodes: dict[NodeId, GraphNode] = {}
        self._edges: list[GraphEdge] = []
        self._next_auto_id = 0
        self.output_node_id: NodeId | None = None
        """The node whose output is returned from `execute()`. None = use the
        last node in topological order, which matches single-sink behaviour."""

    # ------------------------------------------------------------------ accessors

    @property
    def nodes(self) -> list[GraphNode]:
        return list(self._nodes.values())

    @property
    def edges(self) -> list[GraphEdge]:
        return list(self._edges)

    def get_node(self, node_id: NodeId) -> GraphNode:
        return self._nodes[node_id]

    def __len__(self) -> int:
        return len(self._nodes)

    # ------------------------------------------------------------------ mutation

    def add_node(
        self,
        spec: OperationSpec,
        params: dict[str, Any] | None = None,
        position: tuple[float, float] = (0.0, 0.0),
        node_id: NodeId | None = None,
    ) -> GraphNode:
        if node_id is None:
            node_id = self._mint_id()
        elif node_id in self._nodes:
            raise ValueError(f"Node id already in use: {node_id!r}")
        node = GraphNode(
            id=node_id, spec=spec, params=dict(params or {}), position=position
        )
        self._nodes[node_id] = node
        return node

    def remove_node(self, node_id: NodeId) -> GraphNode:
        node = self._nodes.pop(node_id)
        self._edges = [
            e for e in self._edges if e.source != node_id and e.target != node_id
        ]
        if self.output_node_id == node_id:
            self.output_node_id = None
        return node

    def add_edge(self, edge: GraphEdge) -> None:
        self._validate_edge(edge)
        self._edges.append(edge)
        try:
            self.topological_order()
        except ValueError:
            self._edges.pop()
            raise ValueError("adding this edge would create a cycle") from None

    def remove_edge(self, edge: GraphEdge) -> None:
        with contextlib.suppress(ValueError):
            self._edges.remove(edge)

    def clear(self) -> None:
        self._nodes.clear()
        self._edges.clear()
        self._next_auto_id = 0
        self.output_node_id = None

    # ------------------------------------------------------------------ topology

    def topological_order(self) -> list[NodeId]:
        """Kahn's algorithm. Raises ValueError if the graph has a cycle."""
        incoming = dict.fromkeys(self._nodes, 0)
        adjacency: dict[NodeId, list[NodeId]] = {nid: [] for nid in self._nodes}
        for edge in self._edges:
            incoming[edge.target] += 1
            adjacency[edge.source].append(edge.target)
        # Preserve insertion order amongst no-incoming nodes for determinism.
        queue: deque[NodeId] = deque(
            nid for nid in self._nodes if incoming[nid] == 0
        )
        order: list[NodeId] = []
        while queue:
            nid = queue.popleft()
            order.append(nid)
            for downstream in adjacency[nid]:
                incoming[downstream] -= 1
                if incoming[downstream] == 0:
                    queue.append(downstream)
        if len(order) != len(self._nodes):
            raise ValueError("graph contains a cycle")
        return order

    # ------------------------------------------------------------------ execute

    def execute(self, image: np.ndarray) -> np.ndarray:
        if not self._nodes:
            return image.copy()
        order = self.topological_order()
        outputs: dict[NodeId, dict[str, np.ndarray]] = {}

        for nid in order:
            node = self._nodes[nid]
            inputs_by_port = self._collect_inputs(nid, outputs)

            input_args = []
            for port in node.spec.input_ports:
                if port in inputs_by_port:
                    input_args.append(inputs_by_port[port])
                else:
                    # Unconnected port → fall back to the graph's input image.
                    input_args.append(image)

            if not node.enabled:
                # Pass-through: forward the first input on every output port.
                fallback = input_args[0] if input_args else image
                outputs[nid] = dict.fromkeys(node.spec.output_ports, fallback)
                continue

            result = node.call(*input_args)

            if len(node.spec.output_ports) == 1:
                outputs[nid] = {node.spec.output_ports[0]: result}  # type: ignore[dict-item]
            else:
                if not isinstance(result, tuple) or len(result) != len(node.spec.output_ports):
                    raise ValueError(
                        f"{node.spec.id} declared {len(node.spec.output_ports)} output "
                        f"ports but its func did not return a matching tuple"
                    )
                outputs[nid] = dict(zip(node.spec.output_ports, result, strict=True))

        terminal_id = self.output_node_id if self.output_node_id is not None else order[-1]
        terminal_node = self._nodes[terminal_id]
        return outputs[terminal_id][terminal_node.spec.output_ports[0]]

    # ------------------------------------------------------------------ helpers

    def _mint_id(self) -> NodeId:
        while True:
            candidate = f"n{self._next_auto_id}"
            self._next_auto_id += 1
            if candidate not in self._nodes:
                return candidate

    def _validate_edge(self, edge: GraphEdge) -> None:
        if edge.source not in self._nodes:
            raise ValueError(f"unknown source node: {edge.source!r}")
        if edge.target not in self._nodes:
            raise ValueError(f"unknown target node: {edge.target!r}")
        src_spec = self._nodes[edge.source].spec
        if edge.source_port not in src_spec.output_ports:
            raise ValueError(
                f"{src_spec.id} has no output port {edge.source_port!r}"
            )
        tgt_spec = self._nodes[edge.target].spec
        if edge.target_port not in tgt_spec.input_ports:
            raise ValueError(
                f"{tgt_spec.id} has no input port {edge.target_port!r}"
            )
        for existing in self._edges:
            if existing.target == edge.target and existing.target_port == edge.target_port:
                raise ValueError(
                    f"input port {edge.target}:{edge.target_port} is already connected"
                )

    def _collect_inputs(
        self,
        node_id: NodeId,
        outputs: dict[NodeId, dict[str, np.ndarray]],
    ) -> dict[str, np.ndarray]:
        return {
            edge.target_port: outputs[edge.source][edge.source_port]
            for edge in self._edges
            if edge.target == node_id
        }
