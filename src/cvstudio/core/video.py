"""VideoSource — thin wrapper around `cv2.VideoCapture` for cameras and files.

Exposes the subset of OpenCV's capture API the rest of the app needs (read,
release, fps, frame count, position) and accepts an injectable
`capture_factory` so unit tests can swap in a fake without touching the OS.
The class deliberately stays UI-free; the QTimer-driven streaming logic that
emits frames into the preview worker lives in `ui/video_feed_controller.py`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import cv2
import numpy as np

CaptureFactory = Callable[..., Any]


class VideoSource:
    def __init__(
        self,
        source: int | str,
        *,
        capture_factory: CaptureFactory | None = None,
    ) -> None:
        factory: CaptureFactory = capture_factory or cv2.VideoCapture
        self._source = source
        self._capture = factory(source)

    def is_open(self) -> bool:
        return bool(self._capture.isOpened())

    def read(self) -> np.ndarray | None:
        """Return the next frame as a BGR uint8 ndarray, or None at EOF."""
        ok, frame = self._capture.read()
        if not ok or frame is None:
            return None
        return np.asarray(frame)

    def release(self) -> None:
        self._capture.release()

    def fps(self) -> float:
        """Source frame rate. Cameras often report 0 — callers should fall
        back to a sensible default (e.g. 30) when they need a tick interval."""
        return float(self._capture.get(cv2.CAP_PROP_FPS) or 0.0)

    def frame_count(self) -> int:
        """Total frames for video files; 0 for live cameras."""
        return int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    def position(self) -> int:
        """Index of the next frame to be read."""
        return int(self._capture.get(cv2.CAP_PROP_POS_FRAMES) or 0)

    def frame_size(self) -> tuple[int, int]:
        width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        return width, height

    @property
    def source(self) -> int | str:
        return self._source


class VideoRecorder:
    """Writes processed frames to disk via `cv2.VideoWriter`.

    The writer is created lazily on the first call to `write` so the recorder
    can pick up the actual frame size and channel count. Frame size mismatches
    on subsequent writes are silently resized to the initial size, which is
    typical when a Resize op sits in the pipeline.
    """

    def __init__(
        self,
        path: str,
        fps: float,
        fourcc: str = "mp4v",
        *,
        writer_factory: Callable[..., Any] | None = None,
    ) -> None:
        self._path = str(path)
        self._fps = max(1.0, float(fps))
        self._fourcc_str = fourcc
        self._factory = writer_factory or cv2.VideoWriter
        self._writer: Any | None = None
        self._size: tuple[int, int] | None = None

    def write(self, frame: np.ndarray) -> None:
        if frame is None or getattr(frame, "size", 0) == 0:
            return
        bgr = _to_bgr_for_writer(frame)
        h, w = bgr.shape[:2]
        if self._writer is None:
            fourcc = cv2.VideoWriter_fourcc(*self._fourcc_str)  # type: ignore[attr-defined]
            self._writer = self._factory(self._path, fourcc, self._fps, (w, h))
            if not self._writer.isOpened():
                raise OSError(
                    f"Could not open video writer for {self._path!r}; check the "
                    f"extension and codec availability ({self._fourcc_str})."
                )
            self._size = (w, h)
        elif self._size is not None and (w, h) != self._size:
            bgr = cv2.resize(bgr, self._size, interpolation=cv2.INTER_AREA)
        self._writer.write(bgr)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def is_open(self) -> bool:
        return self._writer is not None


def _to_bgr_for_writer(frame: np.ndarray) -> np.ndarray:
    """Coerce `frame` to BGR uint8 so VideoWriter can ingest it regardless of
    upstream channel layout."""
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame
