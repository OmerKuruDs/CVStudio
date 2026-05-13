"""MainWindow — assembles the three panels and wires user actions.

Layout:
    +-----------------------------------------------------+
    | File menu                                           |
    +-----------------+---------------------+-------------+
    | OperationCatalog|     ImageView       | Parameter   |
    | (left)          |     (center)        | Panel       |
    |                 |                     | (right)     |
    +-----------------+---------------------+-------------+
    |              PipelineView (bottom)                  |
    +-----------------------------------------------------+

Live preview is debounced (~120 ms) and runs on a worker QThread so the UI
stays responsive even on large images with expensive operations. Every change
that affects pipeline output (params, ordering, enable, add/remove, new
source image) calls `_request_preview` which schedules the next run.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QCloseEvent, QIcon, QKeySequence
from PySide6.QtWidgets import (
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from cvsandbox.core.codegen import generate_python_code
from cvsandbox.core.pipeline import Pipeline
from cvsandbox.core.registry import get_operation
from cvsandbox.core.serialization import load as load_pipeline
from cvsandbox.core.serialization import save as save_pipeline
from cvsandbox.resources import ICON_PATH
from cvsandbox.ui.code_export_dialog import CodeExportDialog
from cvsandbox.ui.histogram_panel import HistogramPanel
from cvsandbox.ui.image_view import ImageViewWidget
from cvsandbox.ui.operation_catalog import OperationCatalog
from cvsandbox.ui.parameter_panel import ParameterPanel
from cvsandbox.ui.pipeline_view import PipelineView
from cvsandbox.ui.pipeline_worker import PipelineRequest, PipelineWorker

DEBOUNCE_MS = 120


class MainWindow(QMainWindow):
    _execute_requested = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("cvsandbox")
        self.setWindowIcon(QIcon(str(ICON_PATH)))
        self.resize(1200, 800)

        self._source_image: np.ndarray | None = None
        self._pipeline = Pipeline()
        self._next_request_id = 0
        self._latest_request_id = -1

        self._image_view = ImageViewWidget(self)
        self._catalog = OperationCatalog(self)
        self._param_panel = ParameterPanel(self)
        self._histogram_panel = HistogramPanel(self)
        self._pipeline_view = PipelineView(self._pipeline, self)

        right_splitter = QSplitter(Qt.Orientation.Vertical, self)
        right_splitter.addWidget(self._param_panel)
        right_splitter.addWidget(self._histogram_panel)
        right_splitter.setStretchFactor(0, 3)
        right_splitter.setStretchFactor(1, 1)

        right_panel = QWidget(self)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(right_splitter)

        top_splitter = QSplitter(self)
        top_splitter.addWidget(self._catalog)
        top_splitter.addWidget(self._image_view)
        top_splitter.addWidget(right_panel)
        top_splitter.setStretchFactor(0, 1)
        top_splitter.setStretchFactor(1, 4)
        top_splitter.setStretchFactor(2, 2)

        central = QWidget(self)
        central_layout = QVBoxLayout(central)
        central_layout.addWidget(top_splitter, 4)
        central_layout.addWidget(self._pipeline_view, 1)
        self.setCentralWidget(central)

        self.setStatusBar(QStatusBar(self))
        self._build_menu()
        self._setup_worker()
        self._setup_debouncer()
        self._wire_signals()

    def _build_menu(self) -> None:
        self._build_file_menu()
        self._build_view_menu()

    def _build_file_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")

        open_image_action = QAction("&Open Image…", self)
        open_image_action.setShortcut(QKeySequence.StandardKey.Open)
        open_image_action.triggered.connect(self._on_open)
        file_menu.addAction(open_image_action)

        file_menu.addSeparator()

        new_pipeline_action = QAction("&New Pipeline", self)
        new_pipeline_action.setShortcut(QKeySequence.StandardKey.New)
        new_pipeline_action.triggered.connect(self._on_new_pipeline)
        file_menu.addAction(new_pipeline_action)

        open_pipeline_action = QAction("Open &Pipeline…", self)
        open_pipeline_action.triggered.connect(self._on_open_pipeline)
        file_menu.addAction(open_pipeline_action)

        save_pipeline_action = QAction("&Save Pipeline As…", self)
        save_pipeline_action.setShortcut(QKeySequence.StandardKey.SaveAs)
        save_pipeline_action.triggered.connect(self._on_save_pipeline)
        file_menu.addAction(save_pipeline_action)

        file_menu.addSeparator()

        export_code_action = QAction("&Export Code…", self)
        export_code_action.setShortcut("Ctrl+E")
        export_code_action.triggered.connect(self._on_export_code)
        file_menu.addAction(export_code_action)

        file_menu.addSeparator()

        quit_action = QAction("&Quit", self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

    def _build_view_menu(self) -> None:
        view_menu = self.menuBar().addMenu("&View")

        self._split_action = QAction("Split: &Before/After", self)
        self._split_action.setCheckable(True)
        self._split_action.setShortcut("Ctrl+B")
        self._split_action.toggled.connect(self._image_view.set_split_enabled)
        view_menu.addAction(self._split_action)

    def _setup_worker(self) -> None:
        self._worker_thread = QThread(self)
        self._worker = PipelineWorker()
        self._worker.moveToThread(self._worker_thread)
        self._execute_requested.connect(self._worker.execute)
        self._worker.result_ready.connect(self._on_worker_result)
        self._worker.failed.connect(self._on_worker_failed)
        self._worker_thread.start()

    def _setup_debouncer(self) -> None:
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(DEBOUNCE_MS)
        self._debounce.timeout.connect(self._dispatch_preview)

    def _wire_signals(self) -> None:
        self._catalog.operation_chosen.connect(self._on_operation_chosen)
        self._pipeline_view.selection_changed.connect(self._on_selection_changed)
        self._pipeline_view.pipeline_changed.connect(self._request_preview)
        self._param_panel.params_changed.connect(self._request_preview)

    def _on_open(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            str(Path.home()),
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)",
        )
        if not path:
            return
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if image is None:
            QMessageBox.warning(self, "Open failed", f"Could not read image: {path}")
            return
        self._source_image = image
        self._image_view.set_before(image)
        self.statusBar().showMessage(f"Loaded {path}  ·  {image.shape}")
        self._request_preview()

    def _on_new_pipeline(self) -> None:
        self._pipeline.clear()
        self._pipeline_view.refresh()
        self._param_panel.set_node(None)
        self._request_preview()

    def _on_open_pipeline(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(
            self,
            "Open Pipeline",
            str(Path.home()),
            "Pipeline (*.cvpipe.json *.json)",
        )
        if not path:
            return
        try:
            load_pipeline(Path(path), self._pipeline)
        except (OSError, ValueError, KeyError) as exc:
            QMessageBox.warning(self, "Load failed", f"Could not load pipeline: {exc}")
            return
        self._pipeline_view.refresh()
        first_node = self._pipeline.nodes[0] if self._pipeline.nodes else None
        self._param_panel.set_node(first_node)
        if first_node is not None:
            self._pipeline_view.select(0)
        self.statusBar().showMessage(f"Loaded pipeline {path}")
        self._request_preview()

    def _on_save_pipeline(self) -> None:
        path, _filter = QFileDialog.getSaveFileName(
            self,
            "Save Pipeline",
            str(Path.home() / "pipeline.cvpipe.json"),
            "Pipeline (*.cvpipe.json *.json)",
        )
        if not path:
            return
        try:
            save_pipeline(self._pipeline, Path(path))
        except OSError as exc:
            QMessageBox.warning(self, "Save failed", f"Could not save pipeline: {exc}")
            return
        self.statusBar().showMessage(f"Saved pipeline to {path}")

    def _on_export_code(self) -> None:
        try:
            code = generate_python_code(self._pipeline)
        except ValueError as exc:
            QMessageBox.warning(self, "Export failed", str(exc))
            return
        dialog = CodeExportDialog(code, self)
        dialog.exec()

    def _on_operation_chosen(self, spec_id: str) -> None:
        spec = get_operation(spec_id)
        node = self._pipeline.add(spec)
        self._pipeline_view.refresh()
        self._pipeline_view.select(len(self._pipeline.nodes) - 1)
        self._param_panel.set_node(node)
        self._request_preview()

    def _on_selection_changed(self, index: int) -> None:
        if 0 <= index < len(self._pipeline.nodes):
            self._param_panel.set_node(self._pipeline.nodes[index])
        else:
            self._param_panel.set_node(None)

    def _request_preview(self) -> None:
        self._debounce.start()  # restarts if already running

    def _dispatch_preview(self) -> None:
        if self._source_image is None:
            self._image_view.set_image(None)
            self._histogram_panel.clear()
            return
        steps = tuple(
            (node.spec.func, dict(node.params)) for node in self._pipeline.nodes if node.enabled
        )
        self._next_request_id += 1
        self._latest_request_id = self._next_request_id
        request = PipelineRequest(
            request_id=self._next_request_id,
            image=self._source_image,
            steps=steps,
        )
        self._execute_requested.emit(request)

    def _on_worker_result(self, request_id: int, image: object) -> None:
        if request_id != self._latest_request_id:
            return  # a newer request has already superseded this one
        if not isinstance(image, np.ndarray):
            return
        self._image_view.set_image(image)
        self._histogram_panel.set_image(image)

    def _on_worker_failed(self, request_id: int, message: str) -> None:
        if request_id != self._latest_request_id:
            return
        self.statusBar().showMessage(f"Pipeline error: {message}")

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 (Qt override)
        self._worker_thread.quit()
        self._worker_thread.wait(2000)
        super().closeEvent(event)
