from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from cvstudio.ui.visualization_panel import (
    HEATMAP_COLORMAPS,
    POINT_CLOUD_TARGET_POINTS,
    SURFACE_MAX_GRID,
    VisualizationPanel,
    _auto_stride,
    _coerce_to_single_channel,
    _downsample_for_surface,
    _heights_to_rgba,
)


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_panel_constructs_with_image_mode(app: QApplication) -> None:
    panel = VisualizationPanel()
    try:
        # The default (and only, for now) mode is 2D Image.
        assert panel._mode_selector.currentText() == VisualizationPanel.MODE_2D_IMAGE
    finally:
        panel.deleteLater()


def test_panel_set_image_with_bgr_does_not_raise(app: QApplication) -> None:
    panel = VisualizationPanel()
    try:
        rng = np.random.default_rng(0)
        img = rng.integers(0, 256, size=(32, 48, 3), dtype=np.uint8)
        panel.set_image(img)
        # After set_image, the last image should be cached for mode switches.
        assert panel._last_image is img
    finally:
        panel.deleteLater()


def test_panel_set_image_with_grayscale_does_not_raise(app: QApplication) -> None:
    panel = VisualizationPanel()
    try:
        img = np.full((20, 30), 128, dtype=np.uint8)
        panel.set_image(img)
        assert panel._last_image is img
    finally:
        panel.deleteLater()


def test_panel_clear_resets_cache(app: QApplication) -> None:
    panel = VisualizationPanel()
    try:
        img = np.zeros((10, 10), dtype=np.uint8)
        panel.set_image(img)
        panel.clear()
        assert panel._last_image is None
    finally:
        panel.deleteLater()


def test_panel_set_image_none_clears(app: QApplication) -> None:
    panel = VisualizationPanel()
    try:
        img = np.zeros((10, 10), dtype=np.uint8)
        panel.set_image(img)
        panel.set_image(None)
        assert panel._last_image is None
    finally:
        panel.deleteLater()


# --------------------------------------------------------------- Heatmap mode


def test_panel_offers_heatmap_mode(app: QApplication) -> None:
    panel = VisualizationPanel()
    try:
        modes = [
            panel._mode_selector.itemText(i)
            for i in range(panel._mode_selector.count())
        ]
        assert VisualizationPanel.MODE_2D_HEATMAP in modes
    finally:
        panel.deleteLater()


def test_colormap_selector_hidden_in_image_mode(app: QApplication) -> None:
    panel = VisualizationPanel()
    panel.show()  # toolbar visibility only resolves after show
    try:
        panel._mode_selector.setCurrentText(VisualizationPanel.MODE_2D_IMAGE)
        assert not panel._colormap_selector.isVisible()
    finally:
        panel.deleteLater()


def test_colormap_selector_shows_in_heatmap_mode(app: QApplication) -> None:
    panel = VisualizationPanel()
    panel.show()
    try:
        panel._mode_selector.setCurrentText(VisualizationPanel.MODE_2D_HEATMAP)
        assert panel._colormap_selector.isVisible()
    finally:
        panel.deleteLater()


def test_heatmap_mode_accepts_bgr_input(app: QApplication) -> None:
    panel = VisualizationPanel()
    try:
        panel._mode_selector.setCurrentText(VisualizationPanel.MODE_2D_HEATMAP)
        rng = np.random.default_rng(0)
        bgr = rng.integers(0, 256, size=(24, 32, 3), dtype=np.uint8)
        panel.set_image(bgr)
        assert panel._last_image is bgr
    finally:
        panel.deleteLater()


def test_heatmap_colormap_change_does_not_raise(app: QApplication) -> None:
    panel = VisualizationPanel()
    try:
        panel._mode_selector.setCurrentText(VisualizationPanel.MODE_2D_HEATMAP)
        for friendly_name in HEATMAP_COLORMAPS:
            panel._colormap_selector.setCurrentText(friendly_name)
    finally:
        panel.deleteLater()


def test_coerce_to_single_channel_grayscale_passthrough() -> None:
    img = np.full((10, 10), 128, dtype=np.uint8)
    out = _coerce_to_single_channel(img)
    assert out is img


def test_coerce_to_single_channel_bgr_to_gray() -> None:
    img = np.full((10, 10, 3), (50, 100, 200), dtype=np.uint8)
    out = _coerce_to_single_channel(img)
    assert out.ndim == 2
    assert out.shape == (10, 10)


def test_mode_switch_rerenders_cached_image(app: QApplication) -> None:
    """After switching modes, the cached image should be applied to the new mode."""
    panel = VisualizationPanel()
    try:
        img = np.full((10, 10), 200, dtype=np.uint8)
        panel.set_image(img)
        panel._mode_selector.setCurrentText(VisualizationPanel.MODE_2D_HEATMAP)
        # We can't easily inspect rendered pixels, but the cache must still
        # be the original input.
        assert panel._last_image is img
    finally:
        panel.deleteLater()


# --------------------------------------------------------------- Surface mode


def test_panel_offers_surface_mode(app: QApplication) -> None:
    panel = VisualizationPanel()
    try:
        modes = [
            panel._mode_selector.itemText(i)
            for i in range(panel._mode_selector.count())
        ]
        assert VisualizationPanel.MODE_3D_SURFACE in modes
    finally:
        panel.deleteLater()


def test_z_slider_hidden_outside_surface_mode(app: QApplication) -> None:
    panel = VisualizationPanel()
    panel.show()
    try:
        panel._mode_selector.setCurrentText(VisualizationPanel.MODE_2D_IMAGE)
        assert not panel._z_slider.isVisible()
        panel._mode_selector.setCurrentText(VisualizationPanel.MODE_2D_HEATMAP)
        assert not panel._z_slider.isVisible()
    finally:
        panel.deleteLater()


def test_z_slider_visible_in_surface_mode(app: QApplication) -> None:
    panel = VisualizationPanel()
    panel.show()
    try:
        panel._mode_selector.setCurrentText(VisualizationPanel.MODE_3D_SURFACE)
        assert panel._z_slider.isVisible()
        # Colormap selector is also relevant in surface mode (height-coloured).
        assert panel._colormap_selector.isVisible()
    finally:
        panel.deleteLater()


def test_surface_mode_accepts_bgr_input(app: QApplication) -> None:
    panel = VisualizationPanel()
    try:
        panel._mode_selector.setCurrentText(VisualizationPanel.MODE_3D_SURFACE)
        rng = np.random.default_rng(2)
        bgr = rng.integers(0, 256, size=(60, 80, 3), dtype=np.uint8)
        panel.set_image(bgr)
        # The mesh item should exist after the first image.
        assert panel._surface_mode._surface is not None
    finally:
        panel.deleteLater()


def test_surface_z_slider_change_does_not_raise(app: QApplication) -> None:
    panel = VisualizationPanel()
    try:
        panel._mode_selector.setCurrentText(VisualizationPanel.MODE_3D_SURFACE)
        img = np.full((24, 32), 128, dtype=np.uint8)
        panel.set_image(img)
        panel._z_slider.setValue(50)  # 0.5×
        panel._z_slider.setValue(200)  # 2.0×
        panel._z_slider.setValue(1)  # 0.01×
    finally:
        panel.deleteLater()


def test_surface_clear_removes_mesh(app: QApplication) -> None:
    panel = VisualizationPanel()
    try:
        panel._mode_selector.setCurrentText(VisualizationPanel.MODE_3D_SURFACE)
        img = np.full((20, 20), 128, dtype=np.uint8)
        panel.set_image(img)
        assert panel._surface_mode._surface is not None
        panel.clear()
        assert panel._surface_mode._surface is None
    finally:
        panel.deleteLater()


# --------------------------------------------------------------- Helpers


def test_downsample_for_surface_keeps_small_images_unchanged() -> None:
    img = np.full((SURFACE_MAX_GRID // 2, SURFACE_MAX_GRID // 2), 50, dtype=np.uint8)
    out = _downsample_for_surface(img)
    assert out is img


def test_downsample_for_surface_shrinks_large_images() -> None:
    img = np.zeros((SURFACE_MAX_GRID * 4, SURFACE_MAX_GRID * 4), dtype=np.uint8)
    out = _downsample_for_surface(img)
    longest = max(out.shape)
    assert longest <= SURFACE_MAX_GRID


def test_heights_to_rgba_returns_rgba_grid() -> None:
    heights = np.linspace(0, 255, 10 * 10, dtype=np.float32).reshape(10, 10)
    rgba = _heights_to_rgba(heights, "viridis")
    assert rgba.shape == (10, 10, 4)
    assert rgba.dtype == np.float32
    # Alpha channel should be fully opaque for a normal colormap.
    assert (rgba[..., 3] > 0).all()


def test_heights_to_rgba_handles_flat_input() -> None:
    """A flat field has range = 0; the function must not divide by zero."""
    flat = np.full((5, 5), 100.0, dtype=np.float32)
    rgba = _heights_to_rgba(flat, "viridis")
    assert rgba.shape == (5, 5, 4)
    # Every pixel should map to the same colour (start of the colormap).
    first = rgba[0, 0]
    assert np.allclose(rgba, first)


# ----------------------------------------------------------- Point cloud mode


def test_panel_offers_point_cloud_mode(app: QApplication) -> None:
    panel = VisualizationPanel()
    try:
        modes = [
            panel._mode_selector.itemText(i)
            for i in range(panel._mode_selector.count())
        ]
        assert VisualizationPanel.MODE_3D_POINT_CLOUD in modes
    finally:
        panel.deleteLater()


def test_point_cloud_mode_shows_z_and_colormap_controls(app: QApplication) -> None:
    panel = VisualizationPanel()
    panel.show()
    try:
        panel._mode_selector.setCurrentText(VisualizationPanel.MODE_3D_POINT_CLOUD)
        assert panel._z_slider.isVisible()
        assert panel._colormap_selector.isVisible()
    finally:
        panel.deleteLater()


def test_point_cloud_accepts_bgr_input(app: QApplication) -> None:
    panel = VisualizationPanel()
    try:
        panel._mode_selector.setCurrentText(VisualizationPanel.MODE_3D_POINT_CLOUD)
        rng = np.random.default_rng(7)
        bgr = rng.integers(0, 256, size=(80, 100, 3), dtype=np.uint8)
        panel.set_image(bgr)
        assert panel._point_cloud_mode._scatter is not None
    finally:
        panel.deleteLater()


def test_point_cloud_clear_removes_scatter(app: QApplication) -> None:
    panel = VisualizationPanel()
    try:
        panel._mode_selector.setCurrentText(VisualizationPanel.MODE_3D_POINT_CLOUD)
        img = np.full((30, 30), 100, dtype=np.uint8)
        panel.set_image(img)
        assert panel._point_cloud_mode._scatter is not None
        panel.clear()
        assert panel._point_cloud_mode._scatter is None
    finally:
        panel.deleteLater()


def test_point_cloud_z_and_colormap_changes_do_not_raise(app: QApplication) -> None:
    panel = VisualizationPanel()
    try:
        panel._mode_selector.setCurrentText(VisualizationPanel.MODE_3D_POINT_CLOUD)
        img = np.full((40, 40), 128, dtype=np.uint8)
        panel.set_image(img)
        for friendly in HEATMAP_COLORMAPS:
            panel._colormap_selector.setCurrentText(friendly)
        panel._z_slider.setValue(20)
        panel._z_slider.setValue(180)
    finally:
        panel.deleteLater()


def test_auto_stride_keeps_small_images_at_stride_one() -> None:
    # Tiny image: well under the target point budget → stride 1.
    h = w = 50
    assert _auto_stride(h, w) == 1


def test_auto_stride_grows_for_large_images() -> None:
    # 1080p disparity: 1920*1080 = 2_073_600 pixels; target ~30k → stride ~9.
    h, w = 1080, 1920
    stride = _auto_stride(h, w)
    # Sanity: the post-stride point count should be near the target.
    points = (h // stride) * (w // stride)
    assert 0 < points <= POINT_CLOUD_TARGET_POINTS * 1.5
    assert stride > 1
