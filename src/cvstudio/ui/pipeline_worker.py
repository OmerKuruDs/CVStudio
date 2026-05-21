"""Background pipeline execution.

The UI thread builds a `PipelineRequest` — an immutable snapshot of the
`Pipeline` (deep-copied so live UI edits cannot race) plus the source image —
and emits it. A `PipelineWorker` running on its own QThread receives the
request, executes the snapshot via ``Pipeline.execute``, and emits the
result. The `request_id` lets the UI thread drop stale results when the user
has already moved on to another parameter change.

Per-node timings are gathered through ``Graph.execute``'s ``on_step_timed``
callback and emitted as a ``dict[NodeId, float]`` so the UI can render a
per-node timing HUD that survives reorders and disabled nodes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QObject, Signal, Slot

from cvstudio.core.graph import NodeId
from cvstudio.core.pipeline import Pipeline, Roi


@dataclass(frozen=True)
class PipelineRequest:
    request_id: int
    image: np.ndarray
    pipeline: Pipeline
    """Snapshot of the pipeline — the caller must deep-copy before constructing
    this request so the worker thread never observes mid-edit state."""
    roi: tuple[int, int, int, int] | None = None
    """(x, y, w, h) in `image`'s coordinate space. When set, overrides
    `pipeline.roi` for this request. None = use the snapshot's own roi (which
    is typically already None)."""
    roi_paste_to: tuple[int, int] | None = None


class PipelineWorker(QObject):
    """Executes pipeline requests sequentially. Lives on a worker QThread.

    Emits ``result_ready(request_id, image, timings)`` on success — ``timings``
    is a ``dict[NodeId, float]`` of per-node seconds for every node whose
    ``func`` actually ran. Emits ``failed(request_id, message)`` on exception.
    """

    result_ready = Signal(int, object, object)
    failed = Signal(int, str)

    @Slot(object)
    def execute(self, request: PipelineRequest) -> None:
        try:
            pipe = request.pipeline
            if request.roi is not None:
                x, y, w, h = request.roi
                pipe.roi = Roi(x=x, y=y, width=w, height=h)
                pipe.roi_paste_to = request.roi_paste_to

            timings: dict[NodeId, float] = {}

            def _record(node_id: NodeId, seconds: float) -> None:
                timings[node_id] = seconds

            result = pipe.execute(request.image, on_step_timed=_record)
            self.result_ready.emit(request.request_id, result, timings)
        except Exception as exc:
            # Surfaced to the UI via the `failed` signal — no need to re-raise.
            self.failed.emit(request.request_id, str(exc))
