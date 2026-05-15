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

from cvstudio.ai import streaming
from cvstudio.core.pipeline import _fit_crop_to_destination, coerce_to_match

OperationCall = tuple[Callable[..., np.ndarray], dict[str, Any], str]
"""(func, params, node_id). `node_id` is the originating pipeline node id —
ops that care about identity (e.g. AI streaming) read it from the
`streaming.current_node()` thread-local set around each call."""


@dataclass(frozen=True)
class PipelineRequest:
    request_id: int
    image: np.ndarray
    steps: tuple[OperationCall, ...]
    roi: tuple[int, int, int, int] | None = None
    """(x, y, w, h) in `image`'s coordinate space. When set, the worker crops
    `image` to this rectangle, runs the steps on the crop, and splices the
    result back into a copy of the original image. None = full-image mode."""
    roi_paste_to: tuple[int, int] | None = None
    """Optional destination (x, y) for the processed crop. When None the crop
    goes back to its source ROI position; when set the original ROI region is
    left untouched and the crop is pasted at this point instead."""


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
            if request.roi is None:
                image, timings = self._run_steps(request.image.copy(), request.steps)
                self.result_ready.emit(request.request_id, image, tuple(timings))
                return
            image, timings = self._run_with_roi(request)
            self.result_ready.emit(request.request_id, image, tuple(timings))
        except Exception as exc:
            # Surfaced to the UI via the `failed` signal — no need to re-raise.
            self.failed.emit(request.request_id, str(exc))

    @staticmethod
    def _run_steps(
        image: np.ndarray, steps: tuple[OperationCall, ...]
    ) -> tuple[np.ndarray, list[float]]:
        current = image
        timings: list[float] = []
        for func, params, node_id in steps:
            t0 = time.perf_counter()
            streaming.set_current_node(node_id)
            try:
                current = func(current, **params)
            finally:
                streaming.set_current_node(None)
            timings.append(time.perf_counter() - t0)
        return current, timings

    def _run_with_roi(self, request: PipelineRequest) -> tuple[np.ndarray, list[float]]:
        assert request.roi is not None
        x, y, w, h = request.roi
        h_img, w_img = request.image.shape[:2]
        x0 = max(0, min(w_img, x))
        y0 = max(0, min(h_img, y))
        x1 = max(0, min(w_img, x + w))
        y1 = max(0, min(h_img, y + h))
        if x1 <= x0 or y1 <= y0:
            return request.image.copy(), []

        crop = request.image[y0:y1, x0:x1].copy()
        processed, timings = self._run_steps(crop, request.steps)
        # Channel count may differ after the steps (e.g. a grayscale conversion
        # inside a BGR ROI). Convert back to the source layout so the splice
        # actually lands instead of falling through to the no-op branch.
        processed = coerce_to_match(processed, request.image)

        dst_x, dst_y = request.roi_paste_to if request.roi_paste_to is not None else (x0, y0)
        result = request.image.copy()
        try:
            paste = _fit_crop_to_destination(processed, request.image.shape, dst_x, dst_y)
            if paste is None:
                return request.image.copy(), timings
            patch, (px, py) = paste
            ph, pw = patch.shape[:2]
            result[py : py + ph, px : px + pw] = patch
        except (ValueError, TypeError):
            return request.image.copy(), timings
        return result, timings
