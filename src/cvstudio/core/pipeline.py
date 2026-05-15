"""Pipeline — facade over a `Graph` that preserves the linear-chain API.

Internally every pipeline is a DAG (see `core/graph.py`). The `Pipeline` class
keeps a `_chain: list[NodeId]` describing the default linear sequence created
by `Pipeline.add()`, and auto-manages the chain edges that connect consecutive
chain nodes through `output_ports[0]` / `input_ports[0]`. The underlying
`Graph` is exposed via `Pipeline.graph` so the UI can add or remove additional
edges — for example feeding a `Blend` node's second input from an earlier
node — without breaking the chain.

`Pipeline.execute` handles ROI cropping / splicing and delegates the actual
node execution to `Graph.execute`, which does a topological sort and gathers
inputs from incoming edges (falling back to the source image for unconnected
input ports).

`PipelineNode` is kept as an alias for `GraphNode` so older test code that
constructs nodes directly still works.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from cvsandbox.core.graph import Graph, GraphEdge, GraphNode, NodeId
from cvsandbox.core.operation import OperationSpec

SOURCE_SPEC = OperationSpec(
    id="source.image",
    name="Source",
    category="Source",
    description=(
        "The loaded input image. Wire this output into any operation's input "
        "port to feed it the original — useful for multi-input ops like Blend "
        "or Apply Mask that need a reference to the unmodified source."
    ),
    parameters=(),
    func=lambda image: image,  # never actually called — Graph.execute short-circuits zero-input nodes
    input_ports=(),
    output_ports=("image",),
)
SOURCE_NODE_ID = "__source__"


def _fit_crop_to_destination(
    patch: np.ndarray,
    target_shape: tuple[int, ...],
    dst_x: int,
    dst_y: int,
) -> tuple[np.ndarray, tuple[int, int]] | None:
    """Clip `patch` so it fits at `(dst_x, dst_y)` inside an image of
    `target_shape`. Returns the (possibly trimmed) patch and the clamped
    destination origin, or None if the destination has no on-image overlap."""
    target_h, target_w = target_shape[:2]
    ph, pw = patch.shape[:2]

    src_x0 = max(0, -dst_x)
    src_y0 = max(0, -dst_y)
    dst_x0 = max(0, dst_x)
    dst_y0 = max(0, dst_y)

    avail_w = max(0, target_w - dst_x0)
    avail_h = max(0, target_h - dst_y0)
    take_w = min(pw - src_x0, avail_w)
    take_h = min(ph - src_y0, avail_h)
    if take_w <= 0 or take_h <= 0:
        return None

    sliced = patch[src_y0 : src_y0 + take_h, src_x0 : src_x0 + take_w]
    return sliced, (dst_x0, dst_y0)


def coerce_to_match(image: np.ndarray, like: np.ndarray) -> np.ndarray:
    """Return `image` reshaped to match `like`'s channel layout — gray ↔ BGR ↔
    BGRA — so it can be assigned back into a slice of `like`. Falls back to the
    untouched input when there is no clean colour-space mapping (the caller's
    splice will then raise and trigger the original-source fallback)."""
    src_2d = image.ndim == 2
    dst_2d = like.ndim == 2
    src_c = 1 if src_2d else image.shape[2]
    dst_c = 1 if dst_2d else like.shape[2]
    if (src_2d, src_c) == (dst_2d, dst_c):
        return image

    if src_2d:
        if dst_c == 3:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if dst_c == 4:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    elif dst_2d:
        if src_c == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if src_c == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        if src_c == 3 and dst_c == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        if src_c == 4 and dst_c == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    return image


@dataclass(frozen=True, slots=True)
class Roi:
    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                f"Roi requires positive width and height, got {self.width}x{self.height}"
            )

    def clipped_to(self, shape: tuple[int, ...]) -> Roi | None:
        """Return a copy clipped to the image's H,W bounds, or None if the
        intersection is empty."""
        h_img, w_img = shape[:2]
        x0 = max(0, min(w_img, self.x))
        y0 = max(0, min(h_img, self.y))
        x1 = max(0, min(w_img, self.x + self.width))
        y1 = max(0, min(h_img, self.y + self.height))
        if x1 <= x0 or y1 <= y0:
            return None
        return Roi(x=x0, y=y0, width=x1 - x0, height=y1 - y0)


PipelineNode = GraphNode
"""Back-compat alias. The pipeline now uses GraphNode instances directly; tests
that still write `PipelineNode(spec=X)` see the same construction surface."""


class Pipeline:
    def __init__(self) -> None:
        self._graph = Graph()
        self._chain: list[NodeId] = []
        self._source_node_id: NodeId = self._graph.add_node(
            SOURCE_SPEC, node_id=SOURCE_NODE_ID
        ).id
        self.roi: Roi | None = None
        self.roi_paste_to: tuple[int, int] | None = None
        """When set together with `roi`, the processed crop is spliced into
        this (x, y) top-left coordinate instead of back at the ROI's own
        position. Ignored when `roi` is None."""

    # ------------------------------------------------------------------ access

    @property
    def graph(self) -> Graph:
        """The underlying DAG. UI code can add or remove edges directly here to
        wire multi-input operations beyond the auto-managed chain."""
        return self._graph

    @property
    def source_node_id(self) -> NodeId:
        return self._source_node_id

    def chain_index_of(self, node_id: NodeId) -> int:
        """Return the chain position of `node_id`, or -1 if it is not a chain
        node (e.g. the Source node)."""
        try:
            return self._chain.index(node_id)
        except ValueError:
            return -1

    @property
    def nodes(self) -> list[GraphNode]:
        """Chain operation nodes only — the implicit Source node is omitted so
        existing iteration patterns keep working."""
        return [self._graph.get_node(nid) for nid in self._chain]

    def __len__(self) -> int:
        return len(self._chain)

    # ------------------------------------------------------------------ mutate

    def add(self, spec: OperationSpec, params: dict[str, Any] | None = None) -> GraphNode:
        node = self._graph.add_node(spec, params=dict(params) if params else {})
        # The first chain op auto-wires from the Source node so the user sees
        # an explicit data-flow path from the original image into their pipeline.
        prev_id = self._chain[-1] if self._chain else self._source_node_id
        prev_spec = self._graph.get_node(prev_id).spec
        if spec.input_ports:
            self._graph.add_edge(
                GraphEdge(
                    source=prev_id,
                    source_port=prev_spec.output_ports[0],
                    target=node.id,
                    target_port=spec.input_ports[0],
                )
            )
        self._chain.append(node.id)
        self._graph.output_node_id = node.id
        return node

    def remove(self, index: int) -> GraphNode:
        if not (0 <= index < len(self._chain)):
            raise IndexError(index)
        node_id = self._chain.pop(index)
        node = self._graph.get_node(node_id)
        self._graph.remove_node(node_id)
        self._rebuild_chain_edges()
        return node

    def move(self, src: int, dst: int) -> None:
        nid = self._chain.pop(src)
        self._chain.insert(dst, nid)
        self._rebuild_chain_edges()

    def reorder(self, permutation: Sequence[int]) -> None:
        """Permute the chain in-place. `permutation` is the new ordering as
        old-index values — e.g. `[2, 0, 1]` says "what is now at position 0 used
        to be at index 2." Raises if it is not a valid permutation."""
        n = len(self._chain)
        order = list(permutation)
        if len(order) != n or sorted(order) != list(range(n)):
            raise ValueError(
                f"reorder() expects a permutation of 0..{n - 1}, got {order!r}"
            )
        self._chain = [self._chain[i] for i in order]
        self._rebuild_chain_edges()

    def clear(self) -> None:
        self._graph.clear()
        self._chain.clear()
        self._source_node_id = self._graph.add_node(
            SOURCE_SPEC, node_id=SOURCE_NODE_ID
        ).id
        self.roi = None
        self.roi_paste_to = None

    # ------------------------------------------------------------------ helpers

    def _chain_edge(self, src: NodeId, tgt: NodeId) -> GraphEdge:
        src_spec = self._graph.get_node(src).spec
        tgt_spec = self._graph.get_node(tgt).spec
        return GraphEdge(
            source=src,
            source_port=src_spec.output_ports[0],
            target=tgt,
            target_port=tgt_spec.input_ports[0],
        )

    def _rebuild_chain_edges(self) -> None:
        """Drop every existing chain-pattern edge (output_ports[0] →
        input_ports[0] between two adjacent nodes in the [Source, chain...]
        sequence) and re-create them from the current order. User-drawn edges
        on other ports survive."""
        extended = [self._source_node_id, *self._chain]
        extended_set = set(extended)
        for edge in list(self._graph.edges):
            if edge.source not in extended_set or edge.target not in extended_set:
                continue
            source_node = self._graph.get_node(edge.source)
            target_node = self._graph.get_node(edge.target)
            if not target_node.spec.input_ports:
                continue  # Source-style node cannot be a target
            if (
                edge.source_port == source_node.spec.output_ports[0]
                and edge.target_port == target_node.spec.input_ports[0]
            ):
                self._graph.remove_edge(edge)

        for i in range(len(extended) - 1):
            chain_edge = self._chain_edge(extended[i], extended[i + 1])
            try:
                self._graph.add_edge(chain_edge)
            except ValueError:
                conflicts = [
                    e
                    for e in self._graph.edges
                    if e.target == chain_edge.target
                    and e.target_port == chain_edge.target_port
                ]
                for e in conflicts:
                    self._graph.remove_edge(e)
                self._graph.add_edge(chain_edge)
        self._graph.output_node_id = (
            self._chain[-1] if self._chain else self._source_node_id
        )

    # ------------------------------------------------------------------ execute

    def execute(self, image: np.ndarray) -> np.ndarray:
        if self.roi is None:
            return self._graph.execute(image.copy())

        clipped = self.roi.clipped_to(image.shape)
        if clipped is None:
            return image.copy()

        crop = image[
            clipped.y : clipped.y + clipped.height,
            clipped.x : clipped.x + clipped.width,
        ].copy()
        processed = self._graph.execute(crop)
        processed = coerce_to_match(processed, image)

        dst_x, dst_y = (
            self.roi_paste_to if self.roi_paste_to is not None else (clipped.x, clipped.y)
        )
        result = image.copy()
        try:
            paste = _fit_crop_to_destination(processed, image.shape, dst_x, dst_y)
            if paste is None:
                return image.copy()
            patch, (px, py) = paste
            ph, pw = patch.shape[:2]
            result[py : py + ph, px : px + pw] = patch
        except (ValueError, TypeError):
            return image.copy()
        return result
