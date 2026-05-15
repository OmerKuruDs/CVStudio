from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from cvstudio.core.video import VideoSource
from cvstudio.ui.video_feed_controller import VideoFeedController


class _FakeCapture:
    def __init__(self, frames: list[np.ndarray], fps: float = 30.0, opened: bool = True) -> None:
        self._frames = list(frames)
        self._fps = fps
        self._opened = opened
        self._pos = 0
        self.released = False

    def isOpened(self) -> bool:
        return self._opened

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._pos >= len(self._frames):
            return False, None
        frame = self._frames[self._pos]
        self._pos += 1
        return True, frame

    def release(self) -> None:
        self.released = True

    def get(self, prop: int) -> float:
        import cv2

        if prop == cv2.CAP_PROP_FPS:
            return self._fps
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(len(self._frames))
        return 0.0


def _frame(value: int) -> np.ndarray:
    return np.full((4, 4, 3), value, dtype=np.uint8)


def _make_source(frames: list[np.ndarray], **kwargs) -> VideoSource:
    fake = _FakeCapture(frames, **kwargs)
    return VideoSource("dummy", capture_factory=lambda _src: fake)


def test_controller_emits_frame_after_starting(qapp: QApplication) -> None:
    received: list[np.ndarray] = []
    controller = VideoFeedController()
    controller.frame_ready.connect(received.append)

    controller.start(_make_source([_frame(7), _frame(8)]))
    controller._tick()  # drive the timer manually for deterministic tests
    qapp.processEvents()

    assert len(received) == 1
    assert int(received[0][0, 0, 0]) == 7
    controller.stop()


def test_controller_drops_frames_while_in_flight(qapp: QApplication) -> None:
    received: list[np.ndarray] = []
    controller = VideoFeedController()
    controller.frame_ready.connect(received.append)
    controller.start(_make_source([_frame(1), _frame(2), _frame(3)]))

    controller._tick()  # frame 1 emitted, in_flight True
    controller._tick()  # in_flight blocks → no emission
    controller._tick()  # still blocked
    qapp.processEvents()
    assert [int(f[0, 0, 0]) for f in received] == [1]

    controller.mark_processed()
    controller._tick()  # now allowed → frame 2
    qapp.processEvents()
    assert [int(f[0, 0, 0]) for f in received] == [1, 2]
    controller.stop()


def test_controller_emits_finished_at_end_of_video(qapp: QApplication) -> None:
    fired: list[None] = []
    controller = VideoFeedController()
    controller.finished.connect(lambda: fired.append(None))
    controller.start(_make_source([_frame(0)]))

    controller._tick()  # emit the single frame
    controller.mark_processed()
    controller._tick()  # source exhausted → finished
    qapp.processEvents()

    assert fired == [None]
    assert controller.is_active() is False


def test_controller_start_rejects_unopened_source(qapp: QApplication) -> None:
    bad_source = _make_source([], opened=False)
    controller = VideoFeedController()
    with pytest.raises(RuntimeError, match="open"):
        controller.start(bad_source)
    assert controller.is_active() is False


def test_controller_stop_releases_capture(qapp: QApplication) -> None:
    fake = _FakeCapture([_frame(0)])
    source = VideoSource("dummy", capture_factory=lambda _src: fake)
    controller = VideoFeedController()
    controller.start(source)
    controller.stop()
    assert fake.released is True
    assert controller.is_active() is False


def test_controller_marks_processed_resets_in_flight(qapp: QApplication) -> None:
    controller = VideoFeedController()
    controller.start(_make_source([_frame(0), _frame(1)]))
    controller._tick()
    assert controller._in_flight is True
    controller.mark_processed()
    assert controller._in_flight is False
    controller.stop()


def test_controller_pause_blocks_new_frames_until_resume(qapp: QApplication) -> None:
    received: list[np.ndarray] = []
    controller = VideoFeedController()
    controller.frame_ready.connect(received.append)
    controller.start(_make_source([_frame(1), _frame(2), _frame(3)]))

    controller._tick()
    controller.mark_processed()
    assert [int(f[0, 0, 0]) for f in received] == [1]

    controller.pause()
    assert controller.is_paused() is True
    # Manual tick after pause must not pull a new frame.
    controller._tick()
    assert [int(f[0, 0, 0]) for f in received] == [1]

    controller.resume()
    assert controller.is_paused() is False
    controller._tick()
    assert [int(f[0, 0, 0]) for f in received] == [1, 2]
    controller.stop()


def test_controller_stop_clears_paused_flag(qapp: QApplication) -> None:
    controller = VideoFeedController()
    controller.start(_make_source([_frame(0)]))
    controller.pause()
    assert controller.is_paused() is True
    controller.stop()
    assert controller.is_paused() is False
    assert controller.is_active() is False


def test_controller_pause_is_noop_without_source(qapp: QApplication) -> None:
    controller = VideoFeedController()
    controller.pause()  # no source → silently no-op
    assert controller.is_paused() is False
    controller.resume()  # likewise
    assert controller.is_paused() is False
