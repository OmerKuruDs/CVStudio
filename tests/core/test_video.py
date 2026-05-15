from __future__ import annotations

import numpy as np
import pytest

from cvstudio.core.video import VideoSource


class _FakeCapture:
    """Minimal stand-in for cv2.VideoCapture used in tests."""

    def __init__(
        self,
        frames: list[np.ndarray] | None = None,
        fps: float = 30.0,
        opened: bool = True,
        size: tuple[int, int] = (320, 240),
    ) -> None:
        self._frames = list(frames or [])
        self._fps = fps
        self._opened = opened
        self._size = size
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
        if prop == cv2.CAP_PROP_POS_FRAMES:
            return float(self._pos)
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._size[0])
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._size[1])
        return 0.0


def _make_frame(value: int, shape: tuple[int, int, int] = (4, 4, 3)) -> np.ndarray:
    return np.full(shape, value, dtype=np.uint8)


def test_video_source_reports_metadata_from_capture() -> None:
    fake = _FakeCapture(
        frames=[_make_frame(10), _make_frame(20)],
        fps=24.0,
        size=(640, 480),
    )
    source = VideoSource("dummy.mp4", capture_factory=lambda _src: fake)
    assert source.is_open()
    assert source.fps() == 24.0
    assert source.frame_count() == 2
    assert source.frame_size() == (640, 480)


def test_video_source_reads_frames_in_order_then_returns_none_at_eof() -> None:
    fake = _FakeCapture(frames=[_make_frame(1), _make_frame(2)])
    source = VideoSource("dummy.mp4", capture_factory=lambda _src: fake)
    a = source.read()
    b = source.read()
    end = source.read()
    assert a is not None and int(a[0, 0, 0]) == 1
    assert b is not None and int(b[0, 0, 0]) == 2
    assert end is None


def test_video_source_release_calls_underlying_capture() -> None:
    fake = _FakeCapture(frames=[_make_frame(0)])
    source = VideoSource("dummy.mp4", capture_factory=lambda _src: fake)
    source.release()
    assert fake.released is True


def test_video_source_failed_capture_reports_closed() -> None:
    fake = _FakeCapture(frames=[], opened=False)
    source = VideoSource("doesnt-exist", capture_factory=lambda _src: fake)
    assert source.is_open() is False


def test_video_source_position_advances_each_read() -> None:
    fake = _FakeCapture(frames=[_make_frame(i) for i in range(3)])
    source = VideoSource("dummy.mp4", capture_factory=lambda _src: fake)
    assert source.position() == 0
    source.read()
    assert source.position() == 1
    source.read()
    assert source.position() == 2


@pytest.fixture
def captured_frames() -> list[np.ndarray]:
    return [_make_frame(i * 10) for i in range(4)]


# ----------------------------------------------------------------- VideoRecorder


class _FakeWriter:
    def __init__(self, path: str, fourcc: int, fps: float, size: tuple[int, int], opened: bool = True) -> None:
        self.path = path
        self.fourcc = fourcc
        self.fps = fps
        self.size = size
        self.frames: list[np.ndarray] = []
        self.closed = False
        self._opened = opened

    def isOpened(self) -> bool:
        return self._opened

    def write(self, frame: np.ndarray) -> None:
        self.frames.append(frame.copy())

    def release(self) -> None:
        self.closed = True


def _writer_factory(records: list[_FakeWriter], *, opened: bool = True):
    def _factory(path, fourcc, fps, size):
        writer = _FakeWriter(path, fourcc, fps, size, opened=opened)
        records.append(writer)
        return writer

    return _factory


def test_video_recorder_initialises_writer_lazily_on_first_frame() -> None:
    from cvstudio.core.video import VideoRecorder

    records: list[_FakeWriter] = []
    rec = VideoRecorder("out.mp4", fps=30.0, writer_factory=_writer_factory(records))
    assert records == []  # writer not built yet

    rec.write(_make_frame(50, shape=(8, 12, 3)))
    assert len(records) == 1
    writer = records[0]
    assert writer.path == "out.mp4"
    assert writer.size == (12, 8)  # (width, height)
    assert writer.fps == 30.0
    assert len(writer.frames) == 1


def test_video_recorder_promotes_grayscale_frames_to_bgr() -> None:
    from cvstudio.core.video import VideoRecorder

    records: list[_FakeWriter] = []
    rec = VideoRecorder("out.mp4", fps=24.0, writer_factory=_writer_factory(records))

    rec.write(np.full((6, 8), 100, dtype=np.uint8))  # grayscale 2D
    written = records[0].frames[0]
    assert written.shape == (6, 8, 3)


def test_video_recorder_resizes_frames_on_size_mismatch() -> None:
    from cvstudio.core.video import VideoRecorder

    records: list[_FakeWriter] = []
    rec = VideoRecorder("out.mp4", fps=24.0, writer_factory=_writer_factory(records))
    rec.write(_make_frame(0, shape=(4, 4, 3)))
    rec.write(_make_frame(50, shape=(10, 10, 3)))  # different size — gets resized
    writer = records[0]
    assert writer.frames[1].shape[:2] == writer.size[::-1]


def test_video_recorder_raises_when_writer_fails_to_open() -> None:
    from cvstudio.core.video import VideoRecorder

    records: list[_FakeWriter] = []
    rec = VideoRecorder(
        "bad.mp4",
        fps=30.0,
        writer_factory=_writer_factory(records, opened=False),
    )
    with pytest.raises(OSError, match="open"):
        rec.write(_make_frame(0, shape=(4, 4, 3)))


def test_video_recorder_close_releases_underlying_writer() -> None:
    from cvstudio.core.video import VideoRecorder

    records: list[_FakeWriter] = []
    rec = VideoRecorder("out.mp4", fps=30.0, writer_factory=_writer_factory(records))
    rec.write(_make_frame(0))
    rec.close()
    assert records[0].closed is True
    assert rec.is_open() is False


def test_video_recorder_ignores_empty_frames() -> None:
    from cvstudio.core.video import VideoRecorder

    records: list[_FakeWriter] = []
    rec = VideoRecorder("out.mp4", fps=30.0, writer_factory=_writer_factory(records))
    rec.write(np.empty((0, 0), dtype=np.uint8))
    # Writer is still uninitialised because the empty frame was skipped.
    assert records == []
