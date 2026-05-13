"""ImageView — display an OpenCV image (np.ndarray) in a QGraphicsView.

OpenCV gives us BGR uint8 arrays; Qt wants RGB (or grayscale). We convert at the
boundary and keep the np.ndarray as the source of truth — the widget is a thin
view, no image state lives here beyond the current QPixmap.

Interactions:
    * mouse wheel               — zoom at cursor (cursor stays anchored)
    * left-mouse drag           — pan
    * double-click              — reset to fit-in-view

Split mode (before/after):
    When `set_split_enabled(True)` is on AND a `set_before(...)` source image
    is available, the view renders the two images **side by side** — before on
    the left, after on the right, separated by a thin vertical strip. Heights
    are normalized to the after image; the before is rescaled (aspect
    preserved) to match.

If the user has manually zoomed/panned, that state is preserved across
`set_image` calls **as long as the new image has the same dimensions** (the
common case during parameter tuning). Different dimensions or no zoom history
re-fit the view.
"""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QMouseEvent, QPixmap, QResizeEvent, QWheelEvent
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView, QWidget

ZOOM_STEP = 1.15
MAX_ZOOM = 40.0
SEPARATOR_COLOR_BGR = (255, 255, 255)
SEPARATOR_WIDTH = 4


class ImageViewWidget(QGraphicsView):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._image_size: tuple[int, int] | None = None  # (w, h) of currently rendered pixmap
        self._user_zoomed = False

        self._before: np.ndarray | None = None
        self._after: np.ndarray | None = None
        self._split_enabled = False

        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setBackgroundBrush(Qt.GlobalColor.darkGray)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)

    # ------------------------------------------------------------------ public API

    def clear(self) -> None:
        self._scene.clear()
        self._pixmap_item = None
        self._image_size = None
        self._user_zoomed = False
        self._before = None
        self._after = None
        self.resetTransform()

    def set_image(self, image: np.ndarray | None) -> None:
        """Set the 'after' (pipeline output) image. Triggers a re-render."""
        self._after = image
        if image is None:
            self.clear()
            return
        self._refresh()

    def set_before(self, image: np.ndarray | None) -> None:
        """Set the 'before' (source) image used in split mode."""
        self._before = image
        if self._split_enabled:
            self._refresh()

    def set_split_enabled(self, enabled: bool) -> None:
        if enabled == self._split_enabled:
            return
        self._split_enabled = enabled
        self._refresh()

    def is_split_enabled(self) -> bool:
        return self._split_enabled

    def reset_view(self) -> None:
        self._user_zoomed = False
        self._fit()

    # ------------------------------------------------------------------ events

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802 (Qt override)
        if self._pixmap_item is None:
            super().wheelEvent(event)
            return
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = ZOOM_STEP if delta > 0 else 1.0 / ZOOM_STEP
        self._apply_zoom(factor)
        event.accept()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802 (Qt override)
        if event.button() == Qt.MouseButton.LeftButton:
            self.reset_view()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802 (Qt override)
        super().resizeEvent(event)
        if not self._user_zoomed:
            self._fit()

    # ------------------------------------------------------------------ rendering

    def _refresh(self) -> None:
        if self._after is None:
            return
        image = (
            _compose_side_by_side(self._before, self._after)
            if self._is_split_active()
            else self._after
        )
        self._render(image)

    def _render(self, image: np.ndarray) -> None:
        pixmap = _ndarray_to_qpixmap(image)
        new_size = (pixmap.width(), pixmap.height())
        preserve_view = self._user_zoomed and self._image_size == new_size

        self._scene.clear()
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._scene.setSceneRect(pixmap.rect().toRectF())
        self._image_size = new_size

        if not preserve_view:
            self._user_zoomed = False
            self._fit()

    def _is_split_active(self) -> bool:
        return self._split_enabled and self._before is not None and self._after is not None

    # ------------------------------------------------------------------ zoom internals

    def _fit(self) -> None:
        if self._pixmap_item is None:
            return
        self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def _current_scale(self) -> float:
        return float(self.transform().m11())

    def _fit_scale(self) -> float:
        if self._pixmap_item is None or self._image_size is None:
            return 1.0
        viewport = self.viewport().size()
        if viewport.width() <= 0 or viewport.height() <= 0:
            return 1.0
        w, h = self._image_size
        return min(viewport.width() / w, viewport.height() / h)

    def _apply_zoom(self, factor: float) -> None:
        current = self._current_scale()
        target = current * factor
        min_scale = self._fit_scale()
        if target < min_scale:
            target = min_scale
            factor = target / current if current else 1.0
            self._user_zoomed = False
        elif target > MAX_ZOOM:
            factor = MAX_ZOOM / current if current else 1.0
        else:
            self._user_zoomed = True
        self.scale(factor, factor)


# ---------------------------------------------------------------------- helpers


def _to_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def _match_height(image: np.ndarray, target_h: int) -> np.ndarray:
    h, w = image.shape[:2]
    if h == target_h:
        return image
    scale = target_h / h
    new_w = max(1, round(w * scale))
    return cv2.resize(image, (new_w, target_h), interpolation=cv2.INTER_LINEAR)


def _compose_side_by_side(before: np.ndarray | None, after: np.ndarray) -> np.ndarray:
    """Place before and after side by side with a thin separator strip."""
    if before is None:
        return after

    after_bgr = _to_bgr(after)
    before_bgr = _to_bgr(before)

    target_h = after_bgr.shape[0]
    before_bgr = _match_height(before_bgr, target_h)

    separator = np.full((target_h, SEPARATOR_WIDTH, 3), SEPARATOR_COLOR_BGR, dtype=np.uint8)
    return np.hstack([before_bgr, separator, after_bgr])


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
