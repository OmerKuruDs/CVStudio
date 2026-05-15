"""Smoke tests for the QSettings-backed UI state persistence.

We use a unique organization name per test so concurrent runs (or
re-runs after a crash) cannot leak state between tests."""

from __future__ import annotations

import uuid

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from cvstudio.operations import load_builtin_operations
from cvstudio.ui.main_window import MainWindow


@pytest.fixture
def isolated_qsettings(qapp: QApplication, monkeypatch: pytest.MonkeyPatch):
    """Point QApplication at a one-shot organization name so the test
    cannot interfere with the user's real settings file."""
    org = f"cvstudio-test-{uuid.uuid4().hex[:8]}"
    qapp.setOrganizationName(org)
    qapp.setApplicationName(org)
    yield org
    # Clear the per-test settings group to keep CI machines tidy.
    QSettings(org, org).clear()


def test_main_window_save_and_restore_downscale(
    qapp: QApplication, isolated_qsettings: str
) -> None:
    load_builtin_operations()

    win = MainWindow()
    win._downscale_enabled = False
    win._downscale_action.setChecked(False)
    win._save_ui_state()
    win.close()

    win2 = MainWindow()
    assert win2._downscale_enabled is False
    assert win2._downscale_action.isChecked() is False
    win2.close()


def test_save_ui_state_writes_geometry_blob(
    qapp: QApplication, isolated_qsettings: str
) -> None:
    """Verify the QSettings keys actually get populated; we don't
    inspect the binary `saveGeometry()` blob (Qt-internal format) but
    can confirm something landed at the right keys."""
    load_builtin_operations()
    win = MainWindow()
    win._save_ui_state()
    settings = QSettings()
    assert settings.value("window/geometry") is not None
    assert settings.value("window/state") is not None
    assert isinstance(settings.value("splitter/top"), list)
    win.close()


def test_settings_handle_string_typed_bool_values(
    qapp: QApplication, isolated_qsettings: str
) -> None:
    """Some platforms (Linux INI backend) serialize bool as "true"/"false";
    the loader must accept both."""
    load_builtin_operations()
    settings = QSettings()
    settings.setValue("view/downscale", "false")

    win = MainWindow()
    assert win._downscale_enabled is False
    win.close()


def test_save_state_round_trips_activity_mode(
    qapp: QApplication, isolated_qsettings: str
) -> None:
    load_builtin_operations()
    win = MainWindow()
    from cvstudio.ui.activity_bar import ActivityBar

    win._activity_bar.set_current_mode(ActivityBar.MODE_AI)
    win._save_ui_state()
    win.close()

    win2 = MainWindow()
    assert win2._activity_bar.current_mode == ActivityBar.MODE_AI
    win2.close()
