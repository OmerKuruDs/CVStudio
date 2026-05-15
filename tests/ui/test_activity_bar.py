from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication

from cvstudio.ui.activity_bar import ActivityBar


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_activity_bar_constructs_with_op_selected(app: QApplication) -> None:
    bar = ActivityBar()
    try:
        assert bar.current_mode == ActivityBar.MODE_OP
        assert not bar.is_collapsed
    finally:
        bar.deleteLater()


def test_activity_bar_emits_mode_changed_when_button_clicked(app: QApplication) -> None:
    bar = ActivityBar()
    received: list[str] = []
    bar.mode_changed.connect(received.append)
    try:
        bar._buttons[ActivityBar.MODE_2D].click()
        assert received == [ActivityBar.MODE_2D]
        assert bar.current_mode == ActivityBar.MODE_2D
    finally:
        bar.deleteLater()


def test_set_current_mode_changes_selection_and_emits(app: QApplication) -> None:
    bar = ActivityBar()
    received: list[str] = []
    bar.mode_changed.connect(received.append)
    try:
        bar.set_current_mode(ActivityBar.MODE_3D)
        assert bar.current_mode == ActivityBar.MODE_3D
        assert ActivityBar.MODE_3D in received
    finally:
        bar.deleteLater()


def test_set_current_mode_ignores_unknown(app: QApplication) -> None:
    bar = ActivityBar()
    try:
        bar.set_current_mode("Bogus")
        assert bar.current_mode == ActivityBar.MODE_OP
    finally:
        bar.deleteLater()


def test_toggle_collapsed_shrinks_and_relabels(app: QApplication) -> None:
    bar = ActivityBar()
    try:
        expanded_width = bar.width()
        assert bar._buttons[ActivityBar.MODE_OP].text() == "Op"
        bar.toggle_collapsed()
        assert bar.is_collapsed
        assert bar.width() < expanded_width
        assert bar._buttons[ActivityBar.MODE_OP].text() == "O"
        # Toggle back.
        bar.toggle_collapsed()
        assert not bar.is_collapsed
        assert bar._buttons[ActivityBar.MODE_OP].text() == "Op"
    finally:
        bar.deleteLater()


def test_mode_buttons_are_mutually_exclusive(app: QApplication) -> None:
    bar = ActivityBar()
    try:
        bar.set_current_mode(ActivityBar.MODE_2D)
        assert bar._buttons[ActivityBar.MODE_2D].isChecked()
        assert not bar._buttons[ActivityBar.MODE_OP].isChecked()
        assert not bar._buttons[ActivityBar.MODE_3D].isChecked()
        bar.set_current_mode(ActivityBar.MODE_3D)
        assert bar._buttons[ActivityBar.MODE_3D].isChecked()
        assert not bar._buttons[ActivityBar.MODE_2D].isChecked()
    finally:
        bar.deleteLater()


def test_activity_bar_has_ai_mode_button(app: QApplication) -> None:
    bar = ActivityBar()
    try:
        assert ActivityBar.MODE_AI in bar._buttons
        assert bar._buttons[ActivityBar.MODE_AI].text() == "AI"
        bar.set_current_mode(ActivityBar.MODE_AI)
        assert bar.current_mode == ActivityBar.MODE_AI
        bar.toggle_collapsed()
        assert bar._buttons[ActivityBar.MODE_AI].text() == "A"
    finally:
        bar.deleteLater()
