"""HistogramPanel — draws per-channel intensity histograms of the current image.

Plain QPainter, no matplotlib. `cv2.calcHist` produces 256-bin float arrays per
channel; we normalize against the per-frame peak so the highest bar of the
loudest channel always reaches the top of the panel.

For BGR input we draw three translucent filled polygons in B/G/R order. For
grayscale we draw a single light-gray polygon. Alpha channels are ignored.
"""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QPainter, QPaintEvent, QPen, QPolygonF
from PySide6.QtWidgets import QSizePolicy, QWidget

BACKGROUND = QColor("#1e1e1e")
PLACEHOLDER_TEXT_COLOR = QColor("#888888")
GRID_COLOR = QColor("#333333")
FILL_ALPHA = 80
MARGIN = 4

_GRAY_COLOR = QColor(200, 200, 200)
_BGR_COLORS = (
    QColor(80, 130, 240),  # blue   — channel 0
    QColor(80, 200, 80),   # green  — channel 1
    QColor(220, 80, 80),   # red    — channel 2
)


class HistogramPanel(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._histograms: list[tuple[np.ndarray, QColor]] = []
        self.setMinimumHeight(120)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

    def set_image(self, image: np.ndarray | None) -> None:
        self._histograms = compute_histograms(image) if image is not None else []
        self.update()

    def clear(self) -> None:
        self.set_image(None)

    # ----------------------------------------------------------------- paint

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 (Qt override)
        painter = QPainter(self)
        rect = self.rect()
        painter.fillRect(rect, BACKGROUND)

        if not self._histograms:
            painter.setPen(PLACEHOLDER_TEXT_COLOR)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "No image")
            return

        plot_w = rect.width() - 2 * MARGIN
        plot_h = rect.height() - 2 * MARGIN
        if plot_w <= 0 or plot_h <= 0:
            return

        peak = max(float(hist.max()) for hist, _ in self._histograms)
        if peak <= 0:
            return

        self._draw_grid(painter, plot_w, plot_h)
        baseline_y = rect.height() - MARGIN
        for hist, color in self._histograms:
            self._draw_channel(painter, hist, color, plot_w, plot_h, baseline_y, peak)

    def _draw_grid(self, painter: QPainter, plot_w: int, plot_h: int) -> None:
        painter.setPen(QPen(GRID_COLOR, 1, Qt.PenStyle.DashLine))
        for i in range(1, 4):
            x = MARGIN + i * plot_w / 4
            painter.drawLine(QPointF(x, MARGIN), QPointF(x, MARGIN + plot_h))

    def _draw_channel(
        self,
        painter: QPainter,
        hist: np.ndarray,
        color: QColor,
        plot_w: int,
        plot_h: int,
        baseline_y: int,
        peak: float,
    ) -> None:
        bins = len(hist)
        if bins < 2:
            return
        polygon = QPolygonF()
        polygon.append(QPointF(MARGIN, baseline_y))
        for i, count in enumerate(hist):
            x = MARGIN + i * plot_w / (bins - 1)
            y = baseline_y - (float(count) / peak) * plot_h
            polygon.append(QPointF(x, y))
        polygon.append(QPointF(MARGIN + plot_w, baseline_y))

        fill = QColor(color)
        fill.setAlpha(FILL_ALPHA)
        painter.setBrush(fill)
        painter.setPen(QPen(color, 1))
        painter.drawPolygon(polygon)


# ---------------------------------------------------------------------- helpers


def compute_histograms(image: np.ndarray) -> list[tuple[np.ndarray, QColor]]:
    """Compute per-channel 256-bin histograms. Alpha is ignored."""
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    if image.ndim == 2:
        hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
        return [(hist, _GRAY_COLOR)]

    if image.ndim == 3:
        channels = min(image.shape[2], 3)
        return [
            (cv2.calcHist([image], [c], None, [256], [0, 256]).flatten(), _BGR_COLORS[c])
            for c in range(channels)
        ]

    raise ValueError(f"Unsupported image shape: {image.shape}")
