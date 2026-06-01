from __future__ import annotations

from typing import Any

import numpy as np
from PySide6.QtCore import QEventLoop, QThread, QTimer
from PySide6.QtWidgets import QApplication

from cvstudio.core.operation import OperationSpec, Parameter
from cvstudio.core.pipeline import Pipeline
from cvstudio.ui.pipeline_worker import PipelineRequest, PipelineWorker


def _add_func(image: np.ndarray, value: int) -> np.ndarray:
    return np.clip(image.astype(np.int32) + value, 0, 255).astype(np.uint8)


def _boom_func(image: np.ndarray) -> np.ndarray:
    del image
    raise RuntimeError("boom")


def _halve_func(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    return image[: h // 2, : w // 2]


ADD = OperationSpec(
    id="test.add",
    name="Add",
    category="Test",
    description="Add a constant value to every pixel (saturating).",
    parameters=(
        Parameter(name="value", kind="int", default=0, min=-255, max=255, step=1),
    ),
    func=_add_func,
)


BOOM = OperationSpec(
    id="test.boom",
    name="Boom",
    category="Test",
    description="Always raises RuntimeError.",
    parameters=(),
    func=_boom_func,
)


HALVE = OperationSpec(
    id="test.halve",
    name="Halve",
    category="Test",
    description="Crop image to its top-left quadrant.",
    parameters=(),
    func=_halve_func,
)


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
        received: list[tuple[int, np.ndarray, dict[str, float]]] = []
        worker.result_ready.connect(
            lambda rid, img, timings: received.append((rid, img, timings))
        )

        pipe = Pipeline()
        n0 = pipe.add(ADD, params={"value": 5})
        n1 = pipe.add(ADD, params={"value": 7})

        image = np.full((4, 4), 10, dtype=np.uint8)
        request = PipelineRequest(request_id=1, image=image, pipeline=pipe)
        worker.execute(request)
        _wait_for(lambda: bool(received))

        assert received[0][0] == 1
        assert int(received[0][1][0, 0]) == 22  # 10 + 5 + 7
        timings = received[0][2]
        # Both add nodes ran; the implicit source node never invokes func, so
        # it should not appear in the timing dict.
        assert set(timings.keys()) == {n0.id, n1.id}
        assert all(t >= 0 for t in timings.values())
    finally:
        _stop(thread)


def test_worker_emits_failed_on_exception(qapp: QApplication) -> None:
    worker, thread = _make_worker_on_thread()
    try:
        errors: list[tuple[int, str]] = []
        worker.failed.connect(lambda rid, msg: errors.append((rid, msg)))

        pipe = Pipeline()
        pipe.add(BOOM)
        request = PipelineRequest(
            request_id=42,
            image=np.zeros((2, 2), dtype=np.uint8),
            pipeline=pipe,
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
        received: list[tuple[np.ndarray, dict[str, float]]] = []
        worker.result_ready.connect(
            lambda _rid, img, timings: received.append((img, timings))
        )

        image = np.full((3, 3), 99, dtype=np.uint8)
        request = PipelineRequest(request_id=0, image=image, pipeline=Pipeline())
        worker.execute(request)
        _wait_for(lambda: bool(received))

        assert np.array_equal(received[0][0], image)
        assert received[0][0] is not image
        assert received[0][1] == {}
    finally:
        _stop(thread)


def test_worker_applies_roi_crop_and_splice(qapp: QApplication) -> None:
    worker, thread = _make_worker_on_thread()
    try:
        received: list[np.ndarray] = []
        worker.result_ready.connect(lambda _rid, img, _t: received.append(img))

        pipe = Pipeline()
        pipe.add(ADD, params={"value": 50})

        image = np.full((10, 10), 100, dtype=np.uint8)
        request = PipelineRequest(
            request_id=1,
            image=image,
            pipeline=pipe,
            roi=(2, 2, 4, 4, 0.0),
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

        pipe = Pipeline()
        pipe.add(HALVE)

        image = np.full((10, 10), 100, dtype=np.uint8)
        request = PipelineRequest(
            request_id=2,
            image=image,
            pipeline=pipe,
            roi=(2, 2, 4, 4, 0.0),
        )
        worker.execute(request)
        _wait_for(lambda: bool(received))

        out = received[0]
        assert out.shape == image.shape
        assert np.array_equal(out, image)
    finally:
        _stop(thread)


def test_worker_per_node_timings_keyed_by_node_id(qapp: QApplication) -> None:
    worker, thread = _make_worker_on_thread()
    try:
        received: list[dict[str, float]] = []
        worker.result_ready.connect(lambda _rid, _img, timings: received.append(timings))

        pipe = Pipeline()
        a = pipe.add(ADD, params={"value": 1})
        b = pipe.add(ADD, params={"value": 2})
        c = pipe.add(ADD, params={"value": 3})

        request = PipelineRequest(
            request_id=7,
            image=np.zeros((4, 4), dtype=np.uint8),
            pipeline=pipe,
        )
        worker.execute(request)
        _wait_for(lambda: bool(received))

        timings = received[0]
        assert set(timings.keys()) == {a.id, b.id, c.id}
        assert all(isinstance(t, float) and t >= 0 for t in timings.values())


    finally:
        _stop(thread)


def test_worker_runs_multi_input_composite_op(qapp: QApplication) -> None:
    """Regression test: PipelineWorker must execute multi-input ops via the
    DAG (Pipeline.execute), not the old linear-chain assumption that fed only
    a single positional input. Previously inpaint / blend / mask_paste raised
    'missing positional argument' through the UI worker."""
    from cvstudio.core.graph import GraphEdge
    from cvstudio.operations.composite import MASK_PASTE
    from cvstudio.operations.threshold import ALL as THRESHOLD_OPS

    binary_threshold = next(spec for spec in THRESHOLD_OPS if spec.id == "threshold.binary")

    worker, thread = _make_worker_on_thread()
    try:
        received: list[np.ndarray] = []
        worker.result_ready.connect(lambda _rid, img, _t: received.append(img))

        # Build: Source -> Threshold; Source -> MaskPaste.dst; Threshold -> MaskPaste.mask
        pipe = Pipeline()
        thr_node = pipe.add(binary_threshold, params={"thresh": 127, "maxval": 255, "inverse": False})
        mp_node = pipe.add(MASK_PASTE, params={"binarize_threshold": 127, "fill_value": 0})
        for edge in list(pipe.graph.edges):
            if edge.target == mp_node.id and edge.target_port == "dst":
                pipe.graph.remove_edge(edge)
        pipe.graph.add_edge(GraphEdge(
            source=pipe.source_node_id, source_port="image",
            target=mp_node.id, target_port="dst",
        ))
        pipe.graph.add_edge(GraphEdge(
            source=thr_node.id, source_port="out",
            target=mp_node.id, target_port="mask",
        ))

        # Image: half bright (>127), half dark (<127). Threshold marks bright
        # pixels white; MaskPaste paints those pixels black on the dst (source).
        image = np.zeros((4, 8), dtype=np.uint8)
        image[:, 4:] = 200  # right half is bright
        request = PipelineRequest(request_id=99, image=image, pipeline=pipe)
        worker.execute(request)
        _wait_for(lambda: bool(received))

        out = received[0]
        # Left half (dark) survived; right half (bright) painted to 0.
        assert int(out[0, 0]) == 0
        assert int(out[0, 7]) == 0
    finally:
        _stop(thread)
