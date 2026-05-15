from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QLabel

from cvsandbox.operations import load_builtin_operations
from cvsandbox.ui.viz_pages import AddOpButton, Viz2DPage, Viz3DPage


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _load_ops() -> None:
    # AddOpButton reads the registry at construction time, so make sure
    # all built-in ops are available before any test creates one.
    load_builtin_operations()


# ----------------------------------------------------------------- AddOpButton


def test_add_op_button_populates_menu_with_categories(app: QApplication) -> None:
    btn = AddOpButton()
    try:
        menu = btn.menu()
        assert menu is not None
        # Each top-level menu item is a category — at least one expected.
        assert menu.actions(), "Add Op popup should contain at least one category"
        # Categories should have submenus.
        first = menu.actions()[0]
        assert first.menu() is not None
    finally:
        btn.deleteLater()


def test_add_op_button_emits_operation_chosen(app: QApplication) -> None:
    btn = AddOpButton()
    received: list[str] = []
    btn.operation_chosen.connect(received.append)
    try:
        # Walk to the first leaf action across submenus.
        first_cat = btn.menu().actions()[0]
        submenu = first_cat.menu()
        assert submenu is not None
        first_leaf = submenu.actions()[0]
        first_leaf.trigger()
        assert len(received) == 1
        assert "." in received[0]  # spec ids are "<category>.<name>"
    finally:
        btn.deleteLater()


# -------------------------------------------------------------------- Viz2DPage


def test_viz2d_page_defaults_to_image_submode(app: QApplication) -> None:
    page = Viz2DPage()
    try:
        assert page._current_submode == Viz2DPage.SUBMODE_IMAGE
        # Heatmap-only controls are hidden in Image submode.
        page.show()
        assert not page._colormap_selector.isVisible()
    finally:
        page.deleteLater()


def test_viz2d_switch_to_heatmap_shows_colormap(app: QApplication) -> None:
    page = Viz2DPage()
    page.show()
    try:
        page._submode_buttons[Viz2DPage.SUBMODE_HEATMAP].click()
        assert page._current_submode == Viz2DPage.SUBMODE_HEATMAP
        assert page._colormap_selector.isVisible()
    finally:
        page.deleteLater()


def test_viz2d_set_image_caches_for_later_submode_switches(app: QApplication) -> None:
    page = Viz2DPage()
    try:
        img = np.full((30, 40, 3), 128, dtype=np.uint8)
        page.set_image(img)
        assert page._last_image is img
        # Switching submode shouldn't lose the cached image.
        page._submode_buttons[Viz2DPage.SUBMODE_HEATMAP].click()
        assert page._last_image is img
    finally:
        page.deleteLater()


def test_viz2d_install_param_panel_hosts_widget(app: QApplication) -> None:
    page = Viz2DPage()
    panel = QLabel("MockParamPanel")
    try:
        page.install_param_panel(panel)
        assert panel.parent() is page._param_slot
    finally:
        page.deleteLater()
        panel.deleteLater()


# -------------------------------------------------------------------- Viz3DPage


def test_viz3d_page_defaults_to_surface_submode(app: QApplication) -> None:
    page = Viz3DPage()
    try:
        assert page._current_submode == Viz3DPage.SUBMODE_SURFACE
    finally:
        page.deleteLater()


def test_viz3d_switch_to_point_cloud_changes_view(app: QApplication) -> None:
    page = Viz3DPage()
    try:
        page._submode_buttons[Viz3DPage.SUBMODE_POINT_CLOUD].click()
        assert page._current_submode == Viz3DPage.SUBMODE_POINT_CLOUD
    finally:
        page.deleteLater()


def test_viz3d_set_image_renders_in_active_submode(app: QApplication) -> None:
    page = Viz3DPage()
    try:
        img = np.full((40, 50), 100, dtype=np.uint8)
        page.set_image(img)
        # Surface mode is default → mesh item should now exist.
        assert page._surface_widget._surface is not None
    finally:
        page.deleteLater()


def test_viz3d_z_slider_and_colormap_change(app: QApplication) -> None:
    page = Viz3DPage()
    try:
        img = np.full((30, 30), 100, dtype=np.uint8)
        page.set_image(img)
        page._z_slider.setValue(40)
        page._colormap_selector.setCurrentText("Inferno")
        # The point cloud widget shares the z scale + colormap signals;
        # switching submodes should preserve them.
        page._submode_buttons[Viz3DPage.SUBMODE_POINT_CLOUD].click()
        assert page._point_cloud_widget._z_scale != 1.0
    finally:
        page.deleteLater()


def test_viz3d_clear_drops_both_widgets(app: QApplication) -> None:
    page = Viz3DPage()
    try:
        img = np.full((40, 40), 50, dtype=np.uint8)
        page.set_image(img)
        page._submode_buttons[Viz3DPage.SUBMODE_POINT_CLOUD].click()  # populate cloud
        page.set_image(img)  # populate cloud
        # Reset
        page.clear()
        assert page._surface_widget._surface is None
        assert page._point_cloud_widget._scatter is None
    finally:
        page.deleteLater()
