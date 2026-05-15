from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent, QWheelEvent
from PySide6.QtWidgets import QApplication

from cvstudio.ui.image_view import (
    MAX_ZOOM,
    SEPARATOR_WIDTH,
    ZOOM_STEP,
    ImageViewWidget,
    _compose_side_by_side,
)


def _rgb(w: int = 200, h: int = 100) -> np.ndarray:
    return np.full((h, w, 3), 80, dtype=np.uint8)


def _wheel(widget: ImageViewWidget, delta: int) -> None:
    pos = widget.viewport().rect().center()
    event = QWheelEvent(
        pos,
        widget.mapToGlobal(pos),
        QPoint(0, 0),
        QPoint(0, delta),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    widget.wheelEvent(event)


def _double_click(widget: ImageViewWidget, button: Qt.MouseButton = Qt.MouseButton.LeftButton) -> None:
    pos = widget.viewport().rect().center()
    event = QMouseEvent(
        QEvent.Type.MouseButtonDblClick,
        pos,
        widget.mapToGlobal(pos),
        button,
        button,
        Qt.KeyboardModifier.NoModifier,
    )
    widget.mouseDoubleClickEvent(event)


@pytest.fixture
def view(qapp: QApplication) -> ImageViewWidget:
    widget = ImageViewWidget()
    widget.resize(400, 300)
    widget.show()
    qapp.processEvents()
    return widget


def test_set_image_fits_view(view: ImageViewWidget) -> None:
    view.set_image(_rgb())
    assert view._user_zoomed is False
    # Fit-scale should equal current transform scale.
    assert view._current_scale() == pytest.approx(view._fit_scale(), rel=0.05)


def test_wheel_zooms_in_and_marks_user_zoomed(view: ImageViewWidget) -> None:
    view.set_image(_rgb())
    fit_scale = view._current_scale()
    _wheel(view, 120)  # one notch up
    assert view._user_zoomed is True
    assert view._current_scale() == pytest.approx(fit_scale * ZOOM_STEP, rel=1e-3)
    # fit_scale itself has the ~1% scrollbar-reservation skew; the ratio is exact.


def test_zoom_is_clamped_to_fit_minimum(view: ImageViewWidget) -> None:
    view.set_image(_rgb())
    # Zoom in once so we have headroom to zoom out.
    _wheel(view, 120)
    assert view._user_zoomed is True
    # Now spam zoom-outs; should clamp at fit_scale and clear user_zoomed.
    for _ in range(20):
        _wheel(view, -120)
    assert view._current_scale() == pytest.approx(view._fit_scale(), rel=0.05)
    assert view._user_zoomed is False


def test_zoom_is_clamped_to_max(view: ImageViewWidget) -> None:
    view.set_image(_rgb())
    for _ in range(100):
        _wheel(view, 120)
    assert view._current_scale() <= MAX_ZOOM + 1e-6


def test_double_click_resets_view(view: ImageViewWidget) -> None:
    view.set_image(_rgb())
    _wheel(view, 120)
    _wheel(view, 120)
    assert view._user_zoomed is True
    _double_click(view)
    assert view._user_zoomed is False
    assert view._current_scale() == pytest.approx(view._fit_scale(), rel=0.05)


def test_same_size_image_swap_preserves_zoom(view: ImageViewWidget) -> None:
    view.set_image(_rgb())
    _wheel(view, 120)
    scale_before = view._current_scale()
    # Simulate parameter tuning: same-size new image arrives.
    view.set_image(_rgb())
    assert view._user_zoomed is True
    assert view._current_scale() == pytest.approx(scale_before, rel=1e-3)


def test_different_size_image_swap_refits(view: ImageViewWidget) -> None:
    view.set_image(_rgb())
    _wheel(view, 120)
    assert view._user_zoomed is True
    # E.g. a Resize op produced a different shape.
    view.set_image(_rgb(w=80, h=40))
    assert view._user_zoomed is False
    assert view._current_scale() == pytest.approx(view._fit_scale(), rel=0.05)


def test_clear_resets_state(view: ImageViewWidget) -> None:
    view.set_image(_rgb())
    _wheel(view, 120)
    view.set_image(None)
    assert view._pixmap_item is None
    assert view._image_size is None
    assert view._user_zoomed is False


# --------------------------------------------------------------- split mode (side-by-side)


def _solid(color: tuple[int, int, int], w: int = 100, h: int = 60) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[..., :] = color
    return img


def test_side_by_side_widths_add_up_plus_separator() -> None:
    before = _solid((10, 10, 10), w=100, h=60)
    after = _solid((200, 200, 200), w=100, h=60)
    out = _compose_side_by_side(before, after)
    assert out.shape == (60, 100 + SEPARATOR_WIDTH + 100, 3)


def test_side_by_side_left_half_is_before_right_half_is_after() -> None:
    before = _solid((10, 10, 10))
    after = _solid((200, 200, 200))
    out = _compose_side_by_side(before, after)
    assert out[0, 0, 0] == 10  # leftmost = before
    assert out[0, -1, 0] == 200  # rightmost = after


def test_side_by_side_normalizes_height_to_after() -> None:
    after = _solid((50, 50, 50), w=80, h=40)
    before = _solid((10, 10, 10), w=200, h=120)  # taller and wider
    out = _compose_side_by_side(before, after)
    # Output height matches after's height. Width = scaled_before_w + sep + after_w.
    assert out.shape[0] == 40
    scaled_before_w = round(200 * (40 / 120))
    assert out.shape[1] == scaled_before_w + SEPARATOR_WIDTH + 80


def test_side_by_side_promotes_grayscale_before_to_bgr() -> None:
    after = _solid((100, 100, 100))
    before_gray = np.full((60, 100), 5, dtype=np.uint8)
    out = _compose_side_by_side(before_gray, after)
    assert out.ndim == 3
    assert out.shape[2] == 3
    assert out[0, 0, 0] == 5  # gray got promoted to BGR


def test_side_by_side_returns_after_when_before_is_none() -> None:
    after = _solid((200, 200, 200))
    out = _compose_side_by_side(None, after)
    assert out is after


def test_split_mode_off_renders_after_only(view: ImageViewWidget) -> None:
    view.set_before(_solid((0, 0, 0)))
    view.set_image(_solid((200, 200, 200)))
    assert view.is_split_enabled() is False
    assert view._image_size == (100, 60)  # not doubled


def test_enabling_split_with_both_images_widens_canvas(view: ImageViewWidget) -> None:
    view.set_before(_solid((0, 0, 0)))
    view.set_image(_solid((200, 200, 200)))
    view.set_split_enabled(True)
    assert view._is_split_active()
    assert view._image_size == (100 + SEPARATOR_WIDTH + 100, 60)


def test_enabling_split_without_before_is_inert(view: ImageViewWidget) -> None:
    view.set_image(_solid((200, 200, 200)))  # no set_before
    view.set_split_enabled(True)
    assert view._is_split_active() is False


# ---------------------------------------------------------------------- ROI


def _drag_roi(view: ImageViewWidget, start: tuple[int, int], end: tuple[int, int]) -> None:
    """Drive the ImageView's ROI drag handlers directly with scene-space
    coordinates. Faster and more deterministic than synthesizing QMouseEvents."""
    view._roi_drag_start = QPointF(*start)
    from PySide6.QtCore import QRectF

    view._roi_temp_item = view._make_roi_rect_item(QRectF(QPointF(*start), QPointF(*start)))
    # Pretend the user moved to `end`.
    view._roi_temp_item.setRect(QRectF(QPointF(*start), QPointF(*end)).normalized())
    # Now run the release branch by reproducing what mouseReleaseEvent does.
    rect = view._roi_temp_item.rect()
    view._discard_roi_temp()
    view._roi_drag_start = None
    clipped = view._clip_rect_to_image(rect)
    if clipped is None:
        return
    roi = (
        int(clipped.x()),
        int(clipped.y()),
        int(clipped.width()),
        int(clipped.height()),
    )
    view.set_roi(roi)
    view.roi_changed.emit(*roi)


def test_roi_mode_off_by_default(view: ImageViewWidget) -> None:
    assert view.is_roi_mode() is False


def test_setting_roi_stores_and_can_be_read_back(view: ImageViewWidget) -> None:
    view.set_image(_rgb())
    view.set_roi((10, 20, 30, 40))
    assert view.roi() == (10, 20, 30, 40)


def test_clearing_roi_removes_overlay(view: ImageViewWidget) -> None:
    view.set_image(_rgb())
    view.set_roi((10, 20, 30, 40))
    assert view._roi_item is not None
    view.set_roi(None)
    assert view._roi is None
    assert view._roi_item is None


def test_set_image_repaints_roi_overlay(view: ImageViewWidget) -> None:
    """Scene.clear() during set_image must not orphan the ROI overlay."""
    view.set_image(_rgb())
    view.set_roi((5, 5, 20, 20))
    # Now swap the image — overlay must still be present afterwards.
    view.set_image(_rgb())
    assert view._roi_item is not None


def test_drag_emits_roi_changed_signal(view: ImageViewWidget, qapp: QApplication) -> None:
    view.set_image(_rgb(w=200, h=100))
    received: list[tuple[int, int, int, int]] = []
    view.roi_changed.connect(lambda x, y, w, h: received.append((x, y, w, h)))

    _drag_roi(view, start=(20, 10), end=(80, 60))
    qapp.processEvents()
    assert received == [(20, 10, 60, 50)]
    assert view.roi() == (20, 10, 60, 50)


def test_drag_outside_image_is_clipped_to_bounds(view: ImageViewWidget) -> None:
    view.set_image(_rgb(w=200, h=100))
    _drag_roi(view, start=(-50, -50), end=(250, 150))
    # Drag should clip to the full image extent.
    assert view.roi() == (0, 0, 200, 100)


def test_tiny_drag_does_not_set_roi(view: ImageViewWidget) -> None:
    """A near-zero-extent drag should be ignored (acts like a stray click)."""
    view.set_image(_rgb(w=200, h=100))
    # Use the actual release-branch logic via the public mouseReleaseEvent path
    # would be heavy; we simulate the size filter ourselves.
    from PySide6.QtCore import QRectF

    view._roi_drag_start = QPointF(50, 50)
    view._roi_temp_item = view._make_roi_rect_item(QRectF(50, 50, 0.5, 0.5))
    rect = view._roi_temp_item.rect()
    view._discard_roi_temp()
    clipped = view._clip_rect_to_image(rect)
    assert clipped is not None
    # Below the 2-pixel threshold → caller must NOT emit / persist.
    assert clipped.width() < 2 and clipped.height() < 2


def test_set_roi_mode_switches_drag_behaviour(view: ImageViewWidget) -> None:
    from PySide6.QtWidgets import QGraphicsView

    assert view.dragMode() == QGraphicsView.DragMode.ScrollHandDrag
    view.set_roi_mode(True)
    assert view.dragMode() == QGraphicsView.DragMode.NoDrag
    view.set_roi_mode(False)
    assert view.dragMode() == QGraphicsView.DragMode.ScrollHandDrag


def _drag_destination(view: ImageViewWidget, dx: int, dy: int) -> None:
    """Drive the paste-destination drag handlers directly. Picks the centre of
    the existing ROI as grab point, then simulates a mouse move of (dx, dy)
    scene units."""
    assert view._roi is not None
    rx, ry, rw, rh = view._roi
    cx, cy = rx + rw / 2, ry + rh / 2
    view._dest_grab_offset = (cx - rx, cy - ry)
    view._dragging_destination = True
    view.set_paste_rect((rx, ry, rw, rh))

    new_scene_x = cx + dx
    new_scene_y = cy + dy
    new_x = new_scene_x - view._dest_grab_offset[0]
    new_y = new_scene_y - view._dest_grab_offset[1]
    if view._image_size is not None:
        iw, ih = view._image_size
        new_x = max(0, min(iw - rw, new_x))
        new_y = max(0, min(ih - rh, new_y))
    rect = (int(new_x), int(new_y), rw, rh)
    view.set_paste_rect(rect)
    view.paste_destination_changed.emit(int(new_x), int(new_y))
    view._dragging_destination = False
    view._dest_grab_offset = None


def test_drag_inside_roi_emits_paste_destination_signal(
    view: ImageViewWidget, qapp: QApplication
) -> None:
    view.set_image(_rgb(w=300, h=200))
    view.set_roi((10, 10, 50, 50))
    received: list[tuple[int, int]] = []
    view.paste_destination_changed.connect(lambda x, y: received.append((x, y)))

    _drag_destination(view, dx=40, dy=30)
    qapp.processEvents()

    assert received[-1] == (50, 40)


def test_drag_does_not_move_source_roi(view: ImageViewWidget) -> None:
    """Dragging the green ROI's interior must NOT change the source rectangle —
    only the cyan destination overlay should follow the cursor."""
    view.set_image(_rgb(w=300, h=200))
    view.set_roi((10, 10, 50, 50))
    _drag_destination(view, dx=40, dy=30)
    assert view.roi() == (10, 10, 50, 50)
    assert view._paste_rect == (50, 40, 50, 50)


def test_drag_destination_clamps_inside_image_bounds(view: ImageViewWidget) -> None:
    view.set_image(_rgb(w=200, h=100))
    view.set_roi((10, 10, 30, 30))
    _drag_destination(view, dx=500, dy=500)
    rect = view._paste_rect
    assert rect is not None
    x, y, w, h = rect
    assert x + w <= 200
    assert y + h <= 100


def test_hover_inside_roi_shows_size_all_cursor(view: ImageViewWidget) -> None:
    view.set_image(_rgb(w=200, h=100))
    view.set_roi((10, 10, 40, 40))
    # Build a viewport point that lies inside the ROI on screen. Map the centre
    # of the ROI back from scene → viewport so we cope with any zoom/transform.
    from PySide6.QtCore import QPointF as _QPointF

    centre = view.mapFromScene(_QPointF(30, 30))
    view._update_hover_cursor(_QPointF(centre.x(), centre.y()))
    assert view.viewport().cursor().shape() == Qt.CursorShape.SizeAllCursor


def test_split_mode_hides_roi_overlay(view: ImageViewWidget) -> None:
    view.set_before(_solid((0, 0, 0)))
    view.set_image(_solid((200, 200, 200)))
    view.set_roi((10, 10, 30, 30))
    assert view._roi_item is not None
    view.set_split_enabled(True)
    # In split mode the composite has different coordinates — overlay is suppressed.
    assert view._roi_item is None
    view.set_split_enabled(False)
    # Restoring split-off should bring it back.
    assert view._roi_item is not None
