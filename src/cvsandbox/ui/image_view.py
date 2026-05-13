"""ImageView — display an OpenCV image (np.ndarray) in a QGraphicsView.

OpenCV gives us BGR uint8 arrays; Qt wants RGB (or grayscale). We convert at the
boundary and keep the np.ndarray as the source of truth — the widget is a thin
view, no image state lives here beyond the current QPixmap.

Aspect ratio is preserved and the pixmap is re-fitted on resize.
"""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap, QResizeEvent
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView, QWidget


class ImageViewWidget(QGraphicsView):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item: QGraphicsPixmapItem | None = None
        self.setRenderHint(self.renderHints())
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setBackgroundBrush(Qt.GlobalColor.darkGray)

    def clear(self) -> None:
        self._scene.clear()
        self._pixmap_item = None

    def set_image(self, image: np.ndarray | None) -> None:
        if image is None:
            self.clear()
            return
        pixmap = _ndarray_to_qpixmap(image)
        self._scene.clear()
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._scene.setSceneRect(pixmap.rect().toRectF())
        self._fit()

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802 (Qt override)
        super().resizeEvent(event)
        self._fit()

    def _fit(self) -> None:
        if self._pixmap_item is None:
            return
        self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)


def _ndarray_to_qpixmap(image: np.ndarray) -> QPixmap:
    """Convert a BGR/BGRA/grayscale uint8 ndarray to a QPixmap (RGB888 / Grayscale8).

    `cv2.cvtColor` allocates a new array, so the QImage references memory we own;
    we then `.copy()` the QImage to detach it from the ndarray's lifetime.
    """
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    if image.ndim == 2:
        height, width = image.shape
        qimage = QImage(image.data, width, height, width, QImage.Format.Format_Grayscale8)
        return QPixmap.fromImage(qimage.copy())

    if image.ndim == 3:
        height, width, channels = image.shape
        if channels == 4:
            rgba = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
            qimage = QImage(rgba.data, width, height, 4 * width, QImage.Format.Format_RGBA8888)
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            qimage = QImage(rgb.data, width, height, 3 * width, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimage.copy())

    raise ValueError(f"Unsupported image shape: {image.shape}")
