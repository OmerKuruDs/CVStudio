"""Pipeline persistence to/from JSON.

`.cvpipe.json` files are versioned. v2 layout (current):

    {
        "version": 2,
        "nodes": [
            {"id": "_n12345abc",
             "spec_id": "filtering.gaussian_blur",
             "params": {"ksize": 5, "sigma_x": 1.0},
             "enabled": true,
             "position": [200.0, 80.0]},
            ...
        ],
        "edges": [
            {"source": "__source__", "source_port": "image",
             "target": "_n12345abc", "target_port": "in"},
            ...
        ],
        "roi": {...} | None,
        "roi_paste_to": [x, y] | None
    }

v1 (legacy, linear-chain-only) is auto-migrated on load by replaying its
node list through `Pipeline.add`, which rebuilds the chain edges to the
implicit Source. Unknown operation ids raise `KeyError`; unknown parameters
raise `ValueError`. Loading is atomic — either every node materializes and
the target pipeline is replaced, or the target is left untouched.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cvstudio.core.graph import GraphEdge
from cvstudio.core.operation import OperationSpec
from cvstudio.core.pipeline import Pipeline, Roi
from cvstudio.core.registry import get_operation

CURRENT_VERSION = 2


def to_dict(pipeline: Pipeline) -> dict[str, Any]:
    source_id = pipeline.source_node_id
    source_node = pipeline.graph.get_node(source_id)
    payload: dict[str, Any] = {
        "version": CURRENT_VERSION,
        "nodes": [
            {
                "id": node.id,
                "spec_id": node.spec.id,
                "params": dict(node.params),
                "enabled": node.enabled,
                "position": list(node.position) if node.position is not None else None,
            }
            # Source is recreated implicitly on load — its persisted position
            # is stored separately below.
            for node in pipeline.graph.nodes
            if node.id != source_id
        ],
        "edges": [
            {
                "source": edge.source,
                "source_port": edge.source_port,
                "target": edge.target,
                "target_port": edge.target_port,
            }
            for edge in pipeline.graph.edges
        ],
    }
    if source_node.position is not None:
        payload["source_position"] = list(source_node.position)
    if pipeline.roi is not None:
        payload["roi"] = {
            "x": pipeline.roi.x,
            "y": pipeline.roi.y,
            "width": pipeline.roi.width,
            "height": pipeline.roi.height,
        }
    if pipeline.roi_paste_to is not None:
        payload["roi_paste_to"] = list(pipeline.roi_paste_to)
    return payload


def from_dict(data: dict[str, Any], into: Pipeline) -> None:
    version = data.get("version")
    if version == 1:
        _load_v1(data, into)
        return
    if version == 2:
        _load_v2(data, into)
        return
    raise ValueError(f"Unsupported pipeline version: {version!r}")


# ----------------------------------------------------------------------- v1


def _load_v1(data: dict[str, Any], into: Pipeline) -> None:
    materialized: list[tuple[OperationSpec, dict[str, Any], bool]] = []
    for raw in data.get("nodes", []):
        spec = get_operation(raw["id"])
        params = dict(raw.get("params", {}))
        unknown = set(params) - set(spec.default_params())
        if unknown:
            raise ValueError(
                f"Unknown parameter(s) for {spec.id}: {sorted(unknown)}"
            )
        materialized.append((spec, params, bool(raw.get("enabled", True))))

    new_roi = _parse_roi(data)
    new_paste = _parse_paste(data)

    into.clear()
    for spec, params, enabled in materialized:
        node = into.add(spec, params)
        node.enabled = enabled
    into.roi = new_roi
    into.roi_paste_to = new_paste


# ----------------------------------------------------------------------- v2


def _load_v2(data: dict[str, Any], into: Pipeline) -> None:
    # Phase 1: materialize and validate. Any failure raises BEFORE we touch
    # `into`, preserving the atomic-load guarantee.
    saved_source_id = data.get("source_node_id", into.source_node_id)
    node_records: list[dict[str, Any]] = []
    for raw in data.get("nodes", []):
        spec = get_operation(raw["spec_id"])
        params = dict(raw.get("params", {}))
        unknown = set(params) - set(spec.default_params())
        if unknown:
            raise ValueError(
                f"Unknown parameter(s) for {spec.id}: {sorted(unknown)}"
            )
        raw_position = raw.get("position")
        position: tuple[float, float] | None = (
            (float(raw_position[0]), float(raw_position[1]))
            if raw_position is not None
            else None
        )
        node_records.append(
            {
                "id": str(raw["id"]),
                "spec": spec,
                "params": params,
                "enabled": bool(raw.get("enabled", True)),
                "position": position,
            }
        )

    edge_records: list[GraphEdge] = []
    for raw in data.get("edges", []):
        edge_records.append(
            GraphEdge(
                source=str(raw["source"]),
                target=str(raw["target"]),
                source_port=str(raw.get("source_port", "out")),
                target_port=str(raw.get("target_port", "in")),
            )
        )

    new_roi = _parse_roi(data)
    new_paste = _parse_paste(data)

    # Phase 2: commit. Drop everything and rebuild the graph manually so we
    # restore *exactly* the saved topology — no chain auto-edges.
    into.clear()
    # Source is freshly recreated by clear(); the saved file referred to it by
    # whatever id it had at save time, so we remap incoming source references.
    source_id_remap = {saved_source_id: into.source_node_id}

    for record in node_records:
        node = into._graph.add_node(
            record["spec"],
            params=record["params"],
            position=record["position"],
            node_id=record["id"],
        )
        node.enabled = record["enabled"]
        into._chain.append(record["id"])

    for edge in edge_records:
        remapped = GraphEdge(
            source=source_id_remap.get(edge.source, edge.source),
            target=source_id_remap.get(edge.target, edge.target),
            source_port=edge.source_port,
            target_port=edge.target_port,
        )
        into._graph.add_edge(remapped)

    into._graph.output_node_id = (
        into._chain[-1] if into._chain else into.source_node_id
    )

    raw_source_pos = data.get("source_position")
    if raw_source_pos is not None:
        into._graph.get_node(into.source_node_id).position = (
            float(raw_source_pos[0]),
            float(raw_source_pos[1]),
        )

    into.roi = new_roi
    into.roi_paste_to = new_paste


# ----------------------------------------------------------------- shared


def _parse_roi(data: dict[str, Any]) -> Roi | None:
    if "roi" in data and data["roi"] is not None:
        raw_roi = data["roi"]
        return Roi(
            x=int(raw_roi["x"]),
            y=int(raw_roi["y"]),
            width=int(raw_roi["width"]),
            height=int(raw_roi["height"]),
        )
    return None


def _parse_paste(data: dict[str, Any]) -> tuple[int, int] | None:
    if data.get("roi_paste_to") is not None:
        raw_paste = data["roi_paste_to"]
        return (int(raw_paste[0]), int(raw_paste[1]))
    return None


def save(pipeline: Pipeline, path: Path) -> None:
    path.write_text(json.dumps(to_dict(pipeline), indent=2), encoding="utf-8")


def load(path: Path, into: Pipeline) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    from_dict(data, into)
