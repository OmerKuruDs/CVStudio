"""BatchDialog — modal UI that runs `execute_batch` on a background thread.

The dialog owns:
    * input / output folder pickers
    * a checkbox to overwrite existing files
    * a suffix line edit for the output filename
    * a progress bar + label
    * Start / Cancel buttons

Threading: a QThread hosts a `_BatchWorker` (`QObject`) whose single slot
calls `execute_batch` with closures that cross-thread-emit progress and
finish signals. Cancellation is a simple bool the worker polls between files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from cvstudio.core.batch import (
    BatchRequest,
    BatchResult,
    ProcessFunc,
    discover_images,
    execute_batch,
)


class _BatchWorker(QObject):
    progress = Signal(int, int, str)  # (completed, total, current_path)
    finished = Signal(object)  # BatchResult

    def __init__(self) -> None:
        super().__init__()
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    @Slot(object)
    def run(self, request: BatchRequest) -> None:
        result = execute_batch(
            request,
            on_progress=lambda done, total, path: self.progress.emit(
                done, total, str(path)
            ),
            cancel_check=lambda: self._cancelled,
        )
        self.finished.emit(result)


class BatchDialog(QDialog):
    def __init__(self, process: ProcessFunc, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Batch Process Folder")
        self.resize(520, 320)
        self._process = process
        self._worker: _BatchWorker | None = None
        self._thread: QThread | None = None
        self._running = False

        self._input_edit = QLineEdit(self)
        self._output_edit = QLineEdit(self)
        self._suffix_edit = QLineEdit(self)
        self._suffix_edit.setText("_processed")
        self._overwrite = QCheckBox("Overwrite existing files", self)

        input_browse = QPushButton("Browse…", self)
        input_browse.clicked.connect(self._pick_input)
        output_browse = QPushButton("Browse…", self)
        output_browse.clicked.connect(self._pick_output)

        input_row = QHBoxLayout()
        input_row.addWidget(self._input_edit, 1)
        input_row.addWidget(input_browse)

        output_row = QHBoxLayout()
        output_row.addWidget(self._output_edit, 1)
        output_row.addWidget(output_browse)

        form = QFormLayout()
        form.addRow("Input folder:", input_row)
        form.addRow("Output folder:", output_row)
        form.addRow("Filename suffix:", self._suffix_edit)
        form.addRow("", self._overwrite)

        self._progress = QProgressBar(self)
        self._progress.setRange(0, 1)
        self._progress.setValue(0)
        self._status_label = QLabel("Pick folders and click Start.", self)
        self._status_label.setWordWrap(True)

        self._buttons = QDialogButtonBox(self)
        self._start_btn = QPushButton("Start", self)
        self._cancel_btn = QPushButton("Cancel", self)
        self._close_btn = QPushButton("Close", self)
        self._cancel_btn.setEnabled(False)
        self._buttons.addButton(self._start_btn, QDialogButtonBox.ButtonRole.ActionRole)
        self._buttons.addButton(self._cancel_btn, QDialogButtonBox.ButtonRole.ActionRole)
        self._buttons.addButton(self._close_btn, QDialogButtonBox.ButtonRole.RejectRole)
        self._start_btn.clicked.connect(self._on_start)
        self._cancel_btn.clicked.connect(self._on_cancel)
        self._close_btn.clicked.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addSpacing(8)
        layout.addWidget(self._progress)
        layout.addWidget(self._status_label)
        layout.addStretch(1)
        layout.addWidget(self._buttons)

    # ------------------------------------------------------------------ pickers

    def _pick_input(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Pick input folder", self._input_edit.text() or str(Path.home()))
        if path:
            self._input_edit.setText(path)
            count = len(discover_images(Path(path)))
            self._status_label.setText(f"{count} image(s) found in input folder.")

    def _pick_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Pick output folder", self._output_edit.text() or str(Path.home()))
        if path:
            self._output_edit.setText(path)

    # ------------------------------------------------------------------ run

    def _on_start(self) -> None:
        if self._running:
            return
        input_dir = Path(self._input_edit.text().strip())
        output_dir = Path(self._output_edit.text().strip())
        if not input_dir.is_dir():
            QMessageBox.warning(self, "Invalid input", "Pick a valid input folder.")
            return
        if not output_dir.parent.exists():
            QMessageBox.warning(self, "Invalid output", "Output folder's parent must exist.")
            return
        suffix = self._suffix_edit.text()
        if (
            input_dir.resolve() == output_dir.resolve()
            and not suffix
            and not self._overwrite.isChecked()
        ):
            QMessageBox.warning(
                self,
                "Output collision",
                "Input and output folders are the same and no suffix is set. "
                "Either set a suffix or enable Overwrite.",
            )
            return

        paths = discover_images(input_dir)
        if not paths:
            QMessageBox.information(self, "Empty folder", "No image files found.")
            return

        request = BatchRequest(
            input_paths=paths,
            output_dir=output_dir,
            process=self._process,
            output_suffix=suffix,
            overwrite=self._overwrite.isChecked(),
        )
        self._progress.setRange(0, len(paths))
        self._progress.setValue(0)
        self._status_label.setText(f"Starting batch of {len(paths)} files…")
        self._set_running(True)
        self._launch_worker(request)

    def _on_cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self._status_label.setText("Cancelling…")

    def _launch_worker(self, request: BatchRequest) -> None:
        self._thread = QThread(self)
        self._worker = _BatchWorker()
        self._worker.moveToThread(self._thread)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        worker: Any = self._worker  # narrow for the closure below
        self._thread.started.connect(lambda: worker.run(request))
        self._thread.start()

    @Slot(int, int, str)
    def _on_progress(self, done: int, total: int, current: str) -> None:
        self._progress.setMaximum(total)
        self._progress.setValue(done)
        name = Path(current).name if current else ""
        self._status_label.setText(f"Processing {done}/{total}: {name}")

    @Slot(object)
    def _on_finished(self, result: object) -> None:
        if not isinstance(result, BatchResult):
            return
        self._set_running(False)
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(2000)
            self._thread = None
        self._worker = None
        self._progress.setValue(self._progress.maximum())

        suffix_msg = ""
        if result.cancelled:
            suffix_msg = " (cancelled)"
        details = (
            f"Done{suffix_msg}. "
            f"Processed: {result.completed}, "
            f"skipped: {result.skipped}, "
            f"failed: {len(result.failures)}."
        )
        self._status_label.setText(details)
        if result.failures:
            preview = "\n".join(f"- {p.name}: {msg}" for p, msg in result.failures[:5])
            QMessageBox.information(
                self,
                "Batch finished",
                f"{details}\n\nFirst failures:\n{preview}",
            )

    def _set_running(self, running: bool) -> None:
        self._running = running
        self._start_btn.setEnabled(not running)
        self._cancel_btn.setEnabled(running)
        self._input_edit.setEnabled(not running)
        self._output_edit.setEnabled(not running)
        self._suffix_edit.setEnabled(not running)
        self._overwrite.setEnabled(not running)

    # ------------------------------------------------------------------ events

    def reject(self) -> None:
        if self._running and self._worker is not None:
            self._worker.cancel()
            if self._thread is not None:
                self._thread.quit()
                self._thread.wait(2000)
        super().reject()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 (Qt override)
        # Don't let Esc close mid-run; require an explicit Cancel.
        if event.key() == Qt.Key.Key_Escape and self._running:
            return
        super().keyPressEvent(event)
