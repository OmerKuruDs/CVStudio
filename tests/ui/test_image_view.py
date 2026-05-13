from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPoint, Qt
from PySide6.QtGui import QMouseEvent, QWheelEvent
from PySide6.QtWidgets import QApplication

from cvsandbox.ui.image_view import (
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
