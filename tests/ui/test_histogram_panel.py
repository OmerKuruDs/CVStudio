from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication

from cvstudio.ui.histogram_panel import (
    _BGR_COLORS,
    _GRAY_COLOR,
    HistogramPanel,
    compute_histograms,
)


def test_compute_histograms_grayscale_yields_one_channel() -> None:
    gray = np.full((4, 4), 100, dtype=np.uint8)
    hists = compute_histograms(gray)
    assert len(hists) == 1
    hist, color = hists[0]
    assert hist.shape == (256,)
    assert hist[100] == 16  # 16 pixels all at intensity 100
    assert color is _GRAY_COLOR


def test_compute_histograms_bgr_yields_three_channels_with_correct_colors() -> None:
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    img[..., 0] = 30  # B
    img[..., 1] = 60  # G
    img[..., 2] = 90  # R
    hists = compute_histograms(img)
    assert len(hists) == 3
    assert hists[0][1] is _BGR_COLORS[0]  # blue
    assert hists[1][1] is _BGR_COLORS[1]  # green
    assert hists[2][1] is _BGR_COLORS[2]  # red
    assert hists[0][0][30] == 16
    assert hists[1][0][60] == 16
    assert hists[2][0][90] == 16


def test_compute_histograms_ignores_alpha_channel() -> None:
    bgra = np.zeros((4, 4, 4), dtype=np.uint8)
    hists = compute_histograms(bgra)
    assert len(hists) == 3  # alpha dropped


def test_compute_histograms_coerces_non_uint8_input() -> None:
    img = np.full((4, 4), 100.0, dtype=np.float32)
    hists = compute_histograms(img)
    assert hists[0][0][100] == 16


@pytest.fixture
def panel(qapp: QApplication) -> HistogramPanel:
    widget = HistogramPanel()
    widget.resize(300, 150)
    widget.show()
    qapp.processEvents()
    return widget


def test_set_image_stores_histograms(panel: HistogramPanel) -> None:
    img = np.full((10, 10, 3), 50, dtype=np.uint8)
    panel.set_image(img)
    assert len(panel._histograms) == 3


def test_clear_empties_histograms(panel: HistogramPanel) -> None:
    panel.set_image(np.zeros((4, 4, 3), dtype=np.uint8))
    panel.clear()
    assert panel._histograms == []


def test_paint_with_no_image_does_not_crash(panel: HistogramPanel, qapp: QApplication) -> None:
    panel.repaint()
    qapp.processEvents()


def test_paint_with_image_does_not_crash(panel: HistogramPanel, qapp: QApplication) -> None:
    panel.set_image(np.random.default_rng(0).integers(0, 255, size=(40, 40, 3), dtype=np.uint8))
    panel.repaint()
    qapp.processEvents()


def test_bgr_colors_are_distinct() -> None:
    """Sanity check on channel color palette so future edits do not silently break overlay."""
    colors = {color.name() for color in _BGR_COLORS}
    assert len(colors) == 3
    assert _GRAY_COLOR.name() not in colors


def test_color_palette_objects_are_qcolor() -> None:
    assert isinstance(_GRAY_COLOR, QColor)
    for color in _BGR_COLORS:
        assert isinstance(color, QColor)
