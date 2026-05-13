"""Background pipeline execution.

The UI thread builds a `PipelineRequest` — an immutable snapshot of (function,
params) pairs plus the source image — and emits it. A `PipelineWorker` running
on its own QThread receives the request, executes the steps, and emits the
result. The `request_id` lets the UI thread drop stale results when the user
has already moved on to another parameter change.

Snapshotting at request time means the worker never touches the live Pipeline,
so user edits on the UI thread are race-free.

Each step is timed via `perf_counter`; the per-step seconds are emitted
alongside the result image so the UI can render a per-operation timing HUD.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from PySide6.QtCore import QObject, Signal, Slot

OperationCall = tuple[Callable[..., np.ndarray], dict[str, Any]]


@dataclass(frozen=True)
class PipelineRequest:
    request_id: int
    image: np.ndarray
    steps: tuple[OperationCall, ...]


class PipelineWorker(QObject):
    """Executes pipeline requests sequentially. Lives on a worker QThread.

    Emits `result_ready(request_id, image, timings)` on success — `timings` is a
    tuple of per-step seconds in the same order as `request.steps`. Emits
    `failed(request_id, message)` on exception.
    """

    result_ready = Signal(int, object, object)
    failed = Signal(int, str)

    @Slot(object)
    def execute(self, request: PipelineRequest) -> None:
        try:
            current = request.image.copy()
            timings: list[float] = []
            for func, params in request.steps:
                t0 = time.perf_counter()
                current = func(current, **params)
                timings.append(time.perf_counter() - t0)
            self.result_ready.emit(request.request_id, current, tuple(timings))
        except Exception as exc:
            # Surfaced to the UI via the `failed` signal — no need to re-raise.
            self.failed.emit(request.request_id, str(exc))
