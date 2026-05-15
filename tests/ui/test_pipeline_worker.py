from __future__ import annotations

from typing import Any

import numpy as np
from PySide6.QtCore import QEventLoop, QThread, QTimer
from PySide6.QtWidgets import QApplication

from cvstudio.ui.pipeline_worker import PipelineRequest, PipelineWorker


def _add(image: np.ndarray, value: int) -> np.ndarray:
    return np.clip(image.astype(np.int32) + value, 0, 255).astype(np.uint8)


def _boom(image: np.ndarray) -> np.ndarray:
    raise RuntimeError("boom")


def _wait_for(condition: Any, timeout_ms: int = 2000) -> None:
    """Spin the Qt event loop until `condition()` is true or `timeout_ms` elapses."""
    loop = QEventLoop()
    timer = QTimer()
    timer.setInterval(10)
    timer.timeout.connect(lambda: loop.quit() if condition() else None)
    timer.start()
    QTimer.singleShot(timeout_ms, loop.quit)
    loop.exec()
    timer.stop()


def _make_worker_on_thread() -> tuple[PipelineWorker, QThread]:
    thread = QThread()
    worker = PipelineWorker()
    worker.moveToThread(thread)
    thread.start()
    return worker, thread


def _stop(thread: QThread) -> None:
    thread.quit()
    thread.wait(2000)


def test_worker_executes_steps_in_order(qapp: QApplication) -> None:
    worker, thread = _make_worker_on_thread()
    try:
        received: list[tuple[int, np.ndarray, tuple[float, ...]]] = []
        worker.result_ready.connect(
            lambda rid, img, timings: received.append((rid, img, timings))
        )

        image = np.full((4, 4), 10, dtype=np.uint8)
        request = PipelineRequest(
            request_id=1,
            image=image,
            steps=((_add, {"value": 5}, "n0"), (_add, {"value": 7}, "n1")),
        )
        worker.execute(request)  # direct call: same thread context for this unit test
        _wait_for(lambda: bool(received))

        assert received[0][0] == 1
        assert int(received[0][1][0, 0]) == 22
        timings = received[0][2]
        assert len(timings) == 2
        assert all(t >= 0 for t in timings)
    finally:
        _stop(thread)


def test_worker_emits_failed_on_exception(qapp: QApplication) -> None:
    worker, thread = _make_worker_on_thread()
    try:
        errors: list[tuple[int, str]] = []
        worker.failed.connect(lambda rid, msg: errors.append((rid, msg)))

        request = PipelineRequest(
            request_id=42,
            image=np.zeros((2, 2), dtype=np.uint8),
            steps=((_boom, {}, "n_boom"),),
        )
        worker.execute(request)
        _wait_for(lambda: bool(errors))

        assert errors[0][0] == 42
        assert "boom" in errors[0][1]
    finally:
        _stop(thread)


def test_empty_pipeline_returns_a_copy(qapp: QApplication) -> None:
    worker, thread = _make_worker_on_thread()
    try:
        received: list[tuple[np.ndarray, tuple[float, ...]]] = []
        worker.result_ready.connect(
            lambda _rid, img, timings: received.append((img, timings))
        )

        image = np.full((3, 3), 99, dtype=np.uint8)
        request = PipelineRequest(request_id=0, image=image, steps=())
        worker.execute(request)
        _wait_for(lambda: bool(received))

        assert np.array_equal(received[0][0], image)
        assert received[0][0] is not image
        assert received[0][1] == ()
    finally:
        _stop(thread)


def test_worker_applies_roi_crop_and_splice(qapp: QApplication) -> None:
    worker, thread = _make_worker_on_thread()
    try:
        received: list[np.ndarray] = []
        worker.result_ready.connect(lambda _rid, img, _t: received.append(img))

        image = np.full((10, 10), 100, dtype=np.uint8)
        request = PipelineRequest(
            request_id=1,
            image=image,
            steps=((_add, {"value": 50}, "n_roi"),),
            roi=(2, 2, 4, 4),
        )
        worker.execute(request)
        _wait_for(lambda: bool(received))

        out = received[0]
        # Inside the ROI: 100 + 50 = 150
        assert int(out[2, 2]) == 150
        assert int(out[5, 5]) == 150
        # Outside the ROI: unchanged source value
        assert int(out[0, 0]) == 100
        assert int(out[9, 9]) == 100
    finally:
        _stop(thread)


def test_worker_with_roi_returns_source_on_shape_change(qapp: QApplication) -> None:
    """If the steps change the crop's shape (e.g. a resize), splice fails and
    the worker hands back the unmodified source rather than a corrupt result."""
    worker, thread = _make_worker_on_thread()
    try:
        received: list[np.ndarray] = []
        worker.result_ready.connect(lambda _rid, img, _t: received.append(img))

        def _halve(image: np.ndarray) -> np.ndarray:
            h, w = image.shape[:2]
            return image[: h // 2, : w // 2]

        image = np.full((10, 10), 100, dtype=np.uint8)
        request = PipelineRequest(
            request_id=2,
            image=image,
            steps=((_halve, {}, "n_halve"),),
            roi=(2, 2, 4, 4),
        )
        worker.execute(request)
        _wait_for(lambda: bool(received))

        out = received[0]
        assert out.shape == image.shape
        assert np.array_equal(out, image)
    finally:
        _stop(thread)


def test_worker_per_step_timings_match_step_count(qapp: QApplication) -> None:
    worker, thread = _make_worker_on_thread()
    try:
        received: list[tuple[float, ...]] = []
        worker.result_ready.connect(lambda _rid, _img, timings: received.append(timings))

        request = PipelineRequest(
            request_id=7,
            image=np.zeros((4, 4), dtype=np.uint8),
            steps=(
                (_add, {"value": 1}, "a"),
                (_add, {"value": 2}, "b"),
                (_add, {"value": 3}, "c"),
            ),
        )
        worker.execute(request)
        _wait_for(lambda: bool(received))

        assert len(received[0]) == 3
        assert all(isinstance(t, float) and t >= 0 for t in received[0])
    finally:
        _stop(thread)
