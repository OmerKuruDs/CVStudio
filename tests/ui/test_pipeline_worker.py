from __future__ import annotations

from typing import Any

import numpy as np
from PySide6.QtCore import QEventLoop, QThread, QTimer
from PySide6.QtWidgets import QApplication

from cvsandbox.ui.pipeline_worker import PipelineRequest, PipelineWorker


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
        received: list[tuple[int, np.ndarray]] = []
        worker.result_ready.connect(lambda rid, img: received.append((rid, img)))

        image = np.full((4, 4), 10, dtype=np.uint8)
        request = PipelineRequest(
            request_id=1,
            image=image,
            steps=((_add, {"value": 5}), (_add, {"value": 7})),
        )
        worker.execute(request)  # direct call: same thread context for this unit test
        _wait_for(lambda: bool(received))

        assert received[0][0] == 1
        assert int(received[0][1][0, 0]) == 22
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
            steps=((_boom, {}),),
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
        received: list[np.ndarray] = []
        worker.result_ready.connect(lambda _rid, img: received.append(img))

        image = np.full((3, 3), 99, dtype=np.uint8)
        request = PipelineRequest(request_id=0, image=image, steps=())
        worker.execute(request)
        _wait_for(lambda: bool(received))

        assert np.array_equal(received[0], image)
        assert received[0] is not image
    finally:
        _stop(thread)
