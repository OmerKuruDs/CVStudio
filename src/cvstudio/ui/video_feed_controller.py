"""VideoFeedController — drives a QTimer that pulls frames from a `VideoSource`
and emits them into MainWindow's preview path.

Backpressure is handled by an in-flight flag: a new frame is only fetched when
the previous one has finished its trip through the worker. The controller does
not own the worker — `mark_processed()` must be called by the result handler
once the pipeline has caught up. If the timer ticks while a frame is still in
flight, the new tick is dropped, which gives a clean "process as fast as the
pipeline can manage" behaviour.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal

from cvstudio.core.video import VideoSource

DEFAULT_FPS = 30.0


class VideoFeedController(QObject):
    frame_ready = Signal(object)
    """Emitted with each new BGR frame ndarray pulled from the source."""

    finished = Signal()
    """Emitted when the source exhausts (typical for video files)."""

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._source: VideoSource | None = None
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._in_flight = False
        self._paused = False

    # ------------------------------------------------------------------ public

    def start(self, source: VideoSource) -> None:
        """Take ownership of `source` and begin streaming frames."""
        self.stop()
        if not source.is_open():
            source.release()
            raise RuntimeError(f"Could not open source: {source.source!r}")
        self._source = source
        fps = source.fps() or DEFAULT_FPS
        interval = max(1, int(1000.0 / fps))
        self._timer.setInterval(interval)
        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()
        if self._source is not None:
            self._source.release()
            self._source = None
        self._in_flight = False
        self._paused = False

    def pause(self) -> None:
        """Halt frame fetches without releasing the source. A frame already in
        flight will still complete; the next tick after `resume()` picks up
        from there."""
        if self._source is None or self._paused:
            return
        self._timer.stop()
        self._paused = True

    def resume(self) -> None:
        if self._source is None or not self._paused:
            return
        self._paused = False
        self._timer.start()

    def is_active(self) -> bool:
        return self._source is not None

    def is_paused(self) -> bool:
        return self._paused

    def mark_processed(self) -> None:
        """Call from the result handler once the pipeline has consumed the
        most recent frame, freeing the controller to fetch the next one."""
        self._in_flight = False

    def current_source(self) -> VideoSource | None:
        return self._source

    # ------------------------------------------------------------------ internals

    def _tick(self) -> None:
        if self._in_flight or self._paused or self._source is None:
            return
        frame = self._source.read()
        if frame is None:
            self.stop()
            self.finished.emit()
            return
        self._in_flight = True
        self.frame_ready.emit(np.ascontiguousarray(frame))
