"""Python code generation for pipelines.

Produces a self-contained `process(img)` function that reproduces the pipeline
outside of cvsandbox. Generation walks the graph in topological order, gives
every node its own ``step_N`` variable, and resolves each op's input ports
against the actual incoming edges so multi-input and branching pipelines
export to runnable code.

The Source node maps to the function's input parameter, so for a chain
``Source → Gaussian Blur → Canny`` the body becomes:

    step_0 = img
    step_1 = cv2.GaussianBlur(step_0, (5, 5), 0)
    step_2 = cv2.Canny(step_1, 100, 200)
    return step_2

ROI-bound pipelines wrap the whole DAG body inside a crop / splice block.
A `_coerce_to_match` helper is emitted at module top when ROI is set, so
channel-changing steps (To Grayscale, HSV In-Range Mask, ...) still splice
correctly.
"""

from __future__ import annotations

from typing import Any

from cvsandbox.core.graph import Graph, GraphEdge, NodeId
from cvsandbox.core.pipeline import Pipeline

_HEADER = (
    "import cv2",
    "import numpy as np",
    "",
    "",
)


def generate_python_code(
    pipeline: Pipeline,
    *,
    function_name: str = "process",
) -> str:
    order = pipeline.graph.topological_order()
    var_map: dict[NodeId, str] = {nid: f"step_{i}" for i, nid in enumerate(order)}
    source_id = pipeline.source_node_id

    body_steps = _emit_dag_body(pipeline, order, var_map, source_id)

    output_id = (
        pipeline.graph.output_node_id
        if pipeline.graph.output_node_id is not None
        else order[-1]
    )
    output_var = var_map[output_id]

    helpers: list[str] = []
    if pipeline.roi is None:
        body = _wrap_function_body(body_steps, output_var)
    else:
        body = _wrap_function_body_with_roi(
            body_steps, output_var, pipeline.roi, pipeline.roi_paste_to
        )
    # Any reference to `_coerce_to_match` in the body — composite ops include
    # one, and the ROI wrapper uses it for the splice — pulls in the helper.
    if any("_coerce_to_match" in line for line in body):
        helpers = _COERCE_HELPER_LINES

    lines = [
        *_HEADER,
        *helpers,
        f"def {function_name}(img: np.ndarray) -> np.ndarray:",
        *body,
    ]
    return "\n".join(lines) + "\n"


def _emit_dag_body(
    pipeline: Pipeline,
    order: list[NodeId],
    var_map: dict[NodeId, str],
    source_id: NodeId,
) -> list[str]:
    graph = pipeline.graph
    lines: list[str] = [f"{var_map[source_id]} = img  # source"]

    for topo_index, nid in enumerate(order):
        if nid == source_id:
            continue
        node = graph.get_node(nid)
        output_var = var_map[nid]

        # Build the input-variable tuple in the order the spec declares its
        # input ports. Unconnected ports fall back to the source — matching
        # Graph.execute's runtime semantics.
        input_vars = tuple(
            _input_var_for_port(graph, nid, port, var_map, source_id)
            for port in node.spec.input_ports
        )

        if not node.enabled:
            fallback = input_vars[0] if input_vars else var_map[source_id]
            lines.append(f"# [{topo_index}] {node.spec.name} (disabled)")
            lines.append(f"{output_var} = {fallback}")
            continue

        if node.spec.code_export is None:
            raise ValueError(
                f"Operation {node.spec.id!r} does not support code export"
            )
        op_lines = node.spec.code_export(dict(node.params), input_vars, output_var)
        if not op_lines:
            # Spec produced no code — pass the first input through so downstream
            # references stay valid.
            fallback = input_vars[0] if input_vars else var_map[source_id]
            lines.append(f"{output_var} = {fallback}")
            continue
        lines.append(f"# [{topo_index}] {node.spec.name}")
        lines.extend(op_lines)

    return lines


def _input_var_for_port(
    graph: Graph,
    target_id: NodeId,
    port: str,
    var_map: dict[NodeId, str],
    source_id: NodeId,
) -> str:
    edge = _incoming_edge(graph, target_id, port)
    if edge is None:
        return var_map[source_id]
    return var_map[edge.source]


def _incoming_edge(graph: Graph, target_id: NodeId, port: str) -> GraphEdge | None:
    for edge in graph.edges:
        if edge.target == target_id and edge.target_port == port:
            return edge
    return None


def _wrap_function_body(steps: list[str], output_var: str) -> list[str]:
    if len(steps) == 1:  # only the `source = img` assignment — pipeline is empty
        return ["    # (empty pipeline)", "    return img"]
    return [*(f"    {line}" for line in steps), f"    return {output_var}"]


def _wrap_function_body_with_roi(
    steps: list[str],
    output_var: str,
    roi: Any,
    paste_to: tuple[int, int] | None,
) -> list[str]:
    dest_line = (
        f"    _dx, _dy = {paste_to[0]}, {paste_to[1]}"
        if paste_to is not None
        else "    _dx, _dy = _x, _y"
    )
    return [
        f"    _x, _y, _w, _h = {roi.x}, {roi.y}, {roi.width}, {roi.height}",
        dest_line,
        "    _src = img",
        "    img = _src[_y:_y + _h, _x:_x + _w].copy()",
        *(f"    {line}" for line in steps),
        f"    _result = _coerce_to_match({output_var}, _src)",
        "    _out = _src.copy()",
        "    try:",
        "        _ih, _iw = _result.shape[:2]",
        "        _th, _tw = _out.shape[:2]",
        "        _sx0 = max(0, -_dx); _sy0 = max(0, -_dy)",
        "        _dx0 = max(0, _dx); _dy0 = max(0, _dy)",
        "        _tw_a = max(0, _tw - _dx0); _th_a = max(0, _th - _dy0)",
        "        _take_w = min(_iw - _sx0, _tw_a); _take_h = min(_ih - _sy0, _th_a)",
        "        if _take_w > 0 and _take_h > 0:",
        "            _out[_dy0:_dy0 + _take_h, _dx0:_dx0 + _take_w] = "
        "_result[_sy0:_sy0 + _take_h, _sx0:_sx0 + _take_w]",
        "    except (ValueError, TypeError):",
        "        return _src.copy()",
        "    return _out",
    ]


_COERCE_HELPER_LINES: list[str] = [
    "def _coerce_to_match(image: np.ndarray, like: np.ndarray) -> np.ndarray:",
    '    """Reshape image to match like\'s channel layout (gray <-> BGR <-> BGRA)."""',
    "    _src_2d = image.ndim == 2",
    "    _dst_2d = like.ndim == 2",
    "    _src_c = 1 if _src_2d else image.shape[2]",
    "    _dst_c = 1 if _dst_2d else like.shape[2]",
    "    if (_src_2d, _src_c) == (_dst_2d, _dst_c):",
    "        return image",
    "    if _src_2d:",
    "        if _dst_c == 3:",
    "            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)",
    "        if _dst_c == 4:",
    "            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)",
    "    elif _dst_2d:",
    "        if _src_c == 3:",
    "            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)",
    "        if _src_c == 4:",
    "            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)",
    "    else:",
    "        if _src_c == 3 and _dst_c == 4:",
    "            return cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)",
    "        if _src_c == 4 and _dst_c == 3:",
    "            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)",
    "    return image",
    "",
    "",
]
