"""Modal dialog that shows generated Python code with copy / save actions."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class CodeExportDialog(QDialog):
    def __init__(self, code: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export pipeline as Python code")
        self.resize(720, 520)
        self._code = code

        layout = QVBoxLayout(self)

        self._editor = QPlainTextEdit(self)
        self._editor.setPlainText(code)
        self._editor.setReadOnly(True)
        font = self._editor.font()
        font.setFamily("Consolas")
        self._editor.setFont(font)
        layout.addWidget(self._editor)

        buttons = QDialogButtonBox(self)
        copy_button = QPushButton("Copy", self)
        save_button = QPushButton("Save As…", self)
        close_button = QPushButton("Close", self)
        buttons.addButton(copy_button, QDialogButtonBox.ButtonRole.ActionRole)
        buttons.addButton(save_button, QDialogButtonBox.ButtonRole.ActionRole)
        buttons.addButton(close_button, QDialogButtonBox.ButtonRole.RejectRole)
        layout.addWidget(buttons)

        copy_button.clicked.connect(self._on_copy)
        save_button.clicked.connect(self._on_save)
        close_button.clicked.connect(self.reject)

    def _on_copy(self) -> None:
        clipboard = QGuiApplication.clipboard()
        clipboard.setText(self._code)

    def _on_save(self) -> None:
        path, _filter = QFileDialog.getSaveFileName(
            self,
            "Save Python code",
            str(Path.home() / "pipeline.py"),
            "Python (*.py)",
        )
        if not path:
            return
        try:
            Path(path).write_text(self._code, encoding="utf-8")
        except OSError as exc:
            QMessageBox.warning(self, "Save failed", f"Could not write file: {exc}")
