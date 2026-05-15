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
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QImage,
    QMouseEvent,
    QPen,
    QPixmap,
    QResizeEvent,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QWidget,
)

ZOOM_STEP = 1.15
MAX_ZOOM = 40.0
SEPARATOR_COLOR_BGR = (255, 255, 255)
SEPARATOR_WIDTH = 4
ROI_PEN_COLOR = QColor("#00ff66")
PASTE_PEN_COLOR = QColor("#00ccff")
ROI_MIN_SIDE = 2  # pixels in scene coords — below this we treat the drag as a click


class ImageViewWidget(QGraphicsView):
    roi_changed = Signal(int, int, int, int)
    """Emitted with (x, y, width, height) after the user finishes drawing a
    new region in ROI mode. Empty / tiny drags do not emit."""

    paste_destination_changed = Signal(int, int)
    """Emitted with the new (x, y) top-left of the paste destination when the
    user drags from inside an existing ROI rectangle. The source ROI itself
    is left untouched — drag-to-move repositions the *destination*, not the
    crop source."""

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

        self._roi: tuple[int, int, int, int] | None = None
        self._roi_item: QGraphicsRectItem | None = None
        self._roi_mode = False
        self._roi_drag_start: QPointF | None = None
        self._roi_temp_item: QGraphicsRectItem | None = None
        self._dragging_destination = False
        self._dest_grab_offset: tuple[float, float] | None = None
        self._paste_rect: tuple[int, int, int, int] | None = None
        self._paste_item: QGraphicsRectItem | None = None

        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Cool slate-blue keeps contrast with both dark and bright images while
        # matching the icon's navy ramp.
        self.setBackgroundBrush(QColor("#1a2654"))
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)

    # ------------------------------------------------------------------ public API

    def clear(self) -> None:
        self._scene.clear()
        self._pixmap_item = None
        self._roi_item = None
        self._roi_temp_item = None
        self._paste_item = None
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

    def set_roi_mode(self, enabled: bool) -> None:
        """Enter / leave the rectangle-drawing mode. While on, left-drag draws
        an ROI rectangle instead of panning; cursor becomes a crosshair."""
        self._roi_mode = enabled
        if enabled:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.viewport().setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.viewport().unsetCursor()
            self._discard_roi_temp()

    def is_roi_mode(self) -> bool:
        return self._roi_mode

    def set_roi(self, roi: tuple[int, int, int, int] | None) -> None:
        """Set or clear the persistent ROI overlay (without changing mode)."""
        self._roi = roi
        self._refresh_roi_overlay()

    def roi(self) -> tuple[int, int, int, int] | None:
        return self._roi

    def set_paste_rect(self, rect: tuple[int, int, int, int] | None) -> None:
        """Set or clear the cyan paste-destination overlay (a dotted rectangle
        showing where the processed ROI crop will land)."""
        self._paste_rect = rect
        self._refresh_paste_overlay()

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

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 (Qt override)
        # Priority 1: clicking inside an existing ROI starts a destination drag.
        # The green source rectangle stays put; the cyan paste-destination
        # follows the mouse so the user can "lift" the processed crop and
        # drop it somewhere else on the canvas.
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self._roi is not None
            and not self._split_enabled
            and self._point_inside_roi(event.position())
        ):
            scene_pos = self.mapToScene(event.position().toPoint())
            rx, ry, rw, rh = self._roi
            self._dest_grab_offset = (scene_pos.x() - rx, scene_pos.y() - ry)
            self._dragging_destination = True
            # Seed the cyan overlay so the user gets immediate feedback even
            # before they have moved the cursor.
            self.set_paste_rect((rx, ry, rw, rh))
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        # Priority 2: ROI mode → draw a new rectangle.
        if (
            self._roi_mode
            and event.button() == Qt.MouseButton.LeftButton
            and self._pixmap_item is not None
            and not self._split_enabled
        ):
            self._roi_drag_start = self.mapToScene(event.position().toPoint())
            self._discard_roi_temp()
            self._roi_temp_item = self._make_roi_rect_item(
                QRectF(self._roi_drag_start, self._roi_drag_start)
            )
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 (Qt override)
        if (
            self._dragging_destination
            and self._roi is not None
            and self._dest_grab_offset is not None
        ):
            scene_pos = self.mapToScene(event.position().toPoint())
            gx, gy = self._dest_grab_offset
            new_x = scene_pos.x() - gx
            new_y = scene_pos.y() - gy
            _, _, rw, rh = self._roi
            if self._image_size is not None:
                iw, ih = self._image_size
                new_x = max(0, min(iw - rw, new_x))
                new_y = max(0, min(ih - rh, new_y))
            self.set_paste_rect((int(new_x), int(new_y), rw, rh))
            self.paste_destination_changed.emit(int(new_x), int(new_y))
            event.accept()
            return
        if self._roi_mode and self._roi_drag_start is not None and self._roi_temp_item is not None:
            end = self.mapToScene(event.position().toPoint())
            rect = QRectF(self._roi_drag_start, end).normalized()
            self._roi_temp_item.setRect(rect)
            event.accept()
            return
        # Hover feedback: show the move-cursor when the pointer is inside an
        # existing ROI, even when no button is pressed.
        self._update_hover_cursor(event.position())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802 (Qt override)
        if self._dragging_destination and event.button() == Qt.MouseButton.LeftButton:
            self._dragging_destination = False
            self._dest_grab_offset = None
            self._update_hover_cursor(event.position())
            event.accept()
            return
        if (
            self._roi_mode
            and event.button() == Qt.MouseButton.LeftButton
            and self._roi_temp_item is not None
        ):
            rect = self._roi_temp_item.rect()
            self._discard_roi_temp()
            self._roi_drag_start = None
            clipped = self._clip_rect_to_image(rect)
            if clipped is not None and clipped.width() >= ROI_MIN_SIDE and clipped.height() >= ROI_MIN_SIDE:
                roi = (
                    int(clipped.x()),
                    int(clipped.y()),
                    int(clipped.width()),
                    int(clipped.height()),
                )
                self.set_roi(roi)
                self.roi_changed.emit(*roi)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802 (Qt override)
        if event.button() == Qt.MouseButton.LeftButton and not self._roi_mode:
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
        # Scene.clear() invalidates every QGraphicsItem we keep handles to.
        self._roi_item = None
        self._roi_temp_item = None
        self._paste_item = None
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._scene.setSceneRect(pixmap.rect().toRectF())
        self._image_size = new_size
        self._refresh_roi_overlay()
        self._refresh_paste_overlay()

        if not preserve_view:
            self._user_zoomed = False
            self._fit()

    def _is_split_active(self) -> bool:
        return self._split_enabled and self._before is not None and self._after is not None

    # ------------------------------------------------------------------ ROI internals

    def _make_roi_rect_item(self, rect: QRectF) -> QGraphicsRectItem:
        item = QGraphicsRectItem(rect)
        pen = QPen(ROI_PEN_COLOR, 2, Qt.PenStyle.DashLine)
        pen.setCosmetic(True)  # keep the line width constant across zoom levels
        item.setPen(pen)
        self._scene.addItem(item)
        return item

    def _discard_roi_temp(self) -> None:
        if self._roi_temp_item is not None:
            self._scene.removeItem(self._roi_temp_item)
            self._roi_temp_item = None

    def _refresh_roi_overlay(self) -> None:
        if self._roi_item is not None:
            self._scene.removeItem(self._roi_item)
            self._roi_item = None
        # In split mode the canvas is a side-by-side composite — drawing the
        # overlay on its native coordinates would put it on the wrong half.
        if self._roi is None or self._split_enabled:
            return
        x, y, w, h = self._roi
        self._roi_item = self._make_roi_rect_item(QRectF(x, y, w, h))

    def _refresh_paste_overlay(self) -> None:
        if self._paste_item is not None:
            self._scene.removeItem(self._paste_item)
            self._paste_item = None
        if self._paste_rect is None or self._split_enabled:
            return
        x, y, w, h = self._paste_rect
        item = QGraphicsRectItem(QRectF(x, y, w, h))
        pen = QPen(PASTE_PEN_COLOR, 2, Qt.PenStyle.DotLine)
        pen.setCosmetic(True)
        item.setPen(pen)
        self._scene.addItem(item)
        self._paste_item = item

    def _clip_rect_to_image(self, rect: QRectF) -> QRectF | None:
        if self._image_size is None:
            return None
        w, h = self._image_size
        clipped = rect.intersected(QRectF(0, 0, w, h))
        if clipped.width() <= 0 or clipped.height() <= 0:
            return None
        return clipped

    def _point_inside_roi(self, viewport_point: QPointF) -> bool:
        if self._roi is None:
            return False
        scene = self.mapToScene(viewport_point.toPoint())
        x, y, w, h = self._roi
        return x <= scene.x() < x + w and y <= scene.y() < y + h

    def _update_hover_cursor(self, viewport_point: QPointF) -> None:
        """Restore the cursor that matches the current mode / hover state.
        Called from mouseMove so the user gets immediate feedback when they
        approach the ROI rectangle."""
        if self._dragging_destination:
            return  # the press handler already pinned the cursor
        if self._roi is not None and self._point_inside_roi(viewport_point):
            self.viewport().setCursor(Qt.CursorShape.SizeAllCursor)
        elif self._roi_mode:
            self.viewport().setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.viewport().unsetCursor()

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
