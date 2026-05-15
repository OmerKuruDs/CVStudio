from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication

from cvstudio.ui.code_export_dialog import CodeExportDialog


@pytest.fixture
def sample_code() -> str:
    return "import cv2\n\ndef process(img):\n    return img\n"


def test_dialog_shows_code(qapp: QApplication, sample_code: str) -> None:
    dialog = CodeExportDialog(sample_code)
    try:
        assert dialog._editor.toPlainText() == sample_code
        assert dialog._editor.isReadOnly()
    finally:
        dialog.deleteLater()


def test_copy_button_puts_code_on_clipboard(qapp: QApplication, sample_code: str) -> None:
    dialog = CodeExportDialog(sample_code)
    try:
        clipboard = QGuiApplication.clipboard()
        clipboard.clear()
        dialog._on_copy()
        assert clipboard.text() == sample_code
    finally:
        dialog.deleteLater()


def test_save_writes_file(
    qapp: QApplication,
    sample_code: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "pipeline.py"

    def fake_save_dialog(*_args: object, **_kwargs: object) -> tuple[str, str]:
        return str(target), "Python (*.py)"

    monkeypatch.setattr(
        "cvstudio.ui.code_export_dialog.QFileDialog.getSaveFileName",
        fake_save_dialog,
    )

    dialog = CodeExportDialog(sample_code)
    try:
        dialog._on_save()
    finally:
        dialog.deleteLater()

    assert target.read_text(encoding="utf-8") == sample_code
