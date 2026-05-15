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

import random
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QSettings, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QCloseEvent, QIcon, QKeySequence
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from cvstudio.ai.cache_storage import default_cache_path, load_caches, save_caches
from cvstudio.ai.streaming import bus as streaming_bus
from cvstudio.core.codegen import generate_python_code
from cvstudio.core.image_io import read_image
from cvstudio.core.pipeline import Pipeline, Roi
from cvstudio.operations import ai as ai_ops
from cvstudio.core.registry import get_operation
from cvstudio.core.serialization import load as load_pipeline
from cvstudio.core.serialization import save as save_pipeline
from cvstudio.core.video import VideoRecorder, VideoSource
from cvstudio.resources import ICON_PATH
from cvstudio.ui.batch_dialog import BatchDialog
from cvstudio.ui.code_export_dialog import CodeExportDialog
from cvstudio.ui.dataset_page import DatasetPage
from cvstudio.ui.help_dialog import HelpDialog
from cvstudio.ui.histogram_panel import HistogramPanel
from cvstudio.ui.image_action_bar import ImageActionBar
from cvstudio.ui.image_tools_panel import ImageToolsPanel
from cvstudio.ui.image_view import ImageViewWidget
from cvstudio.ui.node_graph_view import NodeGraphView
from cvstudio.ui.operation_catalog import OperationCatalog
from cvstudio.ui.parameter_panel import ParameterPanel
from cvstudio.ui.pipeline_worker import PipelineRequest, PipelineWorker
from cvstudio.ui.activity_bar import ActivityBar
from cvstudio.ui.video_feed_controller import VideoFeedController
from cvstudio.ui.viz_pages import Viz2DPage, Viz3DPage

DEBOUNCE_MS = 120
PREVIEW_MAX_DIM = 1600  # longest-side cap for downscaled-preview mode


def _to_bool(raw: object) -> bool:
    """QSettings on different OSes can return a bool as `bool`, `"true"`/
    `"false"`, or `0`/`1` — normalize so the rest of the code can rely
    on a plain Python bool."""
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(raw, (int, float)):
        return bool(raw)
    return False


def downscale_for_preview(image: np.ndarray, max_dim: int = PREVIEW_MAX_DIM) -> np.ndarray:
    """Shrink `image` so its longest side equals `max_dim`. Returns the input
    unchanged if it is already at-or-below the cap. Uses INTER_AREA, the highest
    quality downscale interpolation in OpenCV."""
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return image
    scale = max_dim / longest
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


class MainWindow(QMainWindow):
    _execute_requested = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("CVStudio")
        self.setWindowIcon(QIcon(str(ICON_PATH)))
        self.resize(1400, 900)

        self._source_image: np.ndarray | None = None
        self._preview_source: np.ndarray | None = None
        self._downscale_enabled = True
        self._pipeline = Pipeline()
        self._next_request_id = 0
        self._latest_request_id = -1

        self._image_view = ImageViewWidget(self)
        self._catalog = OperationCatalog(self)
        self._param_panel = ParameterPanel(self)
        self._histogram_panel = HistogramPanel(self)
        self._pipeline_view = NodeGraphView(self._pipeline, self)

        right_splitter = QSplitter(Qt.Orientation.Vertical, self)
        right_splitter.addWidget(self._param_panel)
        right_splitter.addWidget(self._histogram_panel)
        right_splitter.setStretchFactor(0, 3)
        right_splitter.setStretchFactor(1, 1)

        right_panel = QWidget(self)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(right_splitter)

        self._catalog.setMinimumWidth(160)
        right_panel.setMinimumWidth(280)
        self._image_view.setMinimumWidth(360)

        # Image view + Image tools sidebar + an action bar live in one composite
        # widget. The sidebar tracks the image area on the right; the action bar
        # sits directly underneath so File-menu staples (Open / Save / Record)
        # are always one click away.
        self._image_with_tools = QWidget(self)
        composite_layout = QVBoxLayout(self._image_with_tools)
        composite_layout.setContentsMargins(0, 0, 0, 0)
        composite_layout.setSpacing(0)
        self._image_row = QWidget(self._image_with_tools)
        image_row_layout = QHBoxLayout(self._image_row)
        image_row_layout.setContentsMargins(0, 0, 0, 0)
        image_row_layout.setSpacing(0)
        image_row_layout.addWidget(self._image_view, 1)
        composite_layout.addWidget(self._image_row, 1)
        # `self._tools_panel` and the action bar are wired in after the menu
        # builds its QActions; see `_install_tools_sidebar` and
        # `_install_action_bar`.

        top_splitter = QSplitter(self)
        top_splitter.addWidget(self._catalog)
        top_splitter.addWidget(self._image_with_tools)
        top_splitter.addWidget(right_panel)
        top_splitter.setStretchFactor(0, 0)
        top_splitter.setStretchFactor(1, 1)
        top_splitter.setStretchFactor(2, 0)
        top_splitter.setCollapsible(0, False)
        top_splitter.setCollapsible(1, False)
        top_splitter.setCollapsible(2, False)
        # Initial proportions for a 1400-wide window: catalog 200, image+tools
        # ~860, right column 340. Qt re-honours these against the actual width
        # when the window is first shown.
        top_splitter.setSizes([200, 860, 340])
        self._top_splitter = top_splitter

        # The Op-mode page wraps the original three-column splitter so the
        # historical "Catalog | Image | Param+Histogram" layout is fully
        # preserved when the activity bar is on Op.
        op_page = QWidget(self)
        op_page_layout = QVBoxLayout(op_page)
        op_page_layout.setContentsMargins(0, 0, 0, 0)
        op_page_layout.addWidget(top_splitter)
        self._op_page = op_page
        # Where the ParameterPanel lives when Op mode is active. The
        # widget itself is a child of `right_splitter` above; we keep a
        # reference to its layout so we can re-host the panel here after
        # a viz page borrows it.
        self._param_panel_op_layout = right_splitter

        # 2D / 3D pages — compact layouts that share the same param panel.
        self._viz_2d_page = Viz2DPage(self)
        self._viz_3d_page = Viz3DPage(self)
        for page in (self._viz_2d_page, self._viz_3d_page):
            page.operation_chosen.connect(self._on_operation_chosen)

        self._content_stack = QStackedWidget(self)
        self._content_stack.addWidget(self._op_page)  # index 0 = Op
        self._content_stack.addWidget(self._viz_2d_page)  # index 1 = 2D
        self._content_stack.addWidget(self._viz_3d_page)  # index 2 = 3D

        # Vertical splitter — top is whatever page is active, bottom is the
        # always-on pipeline graph. The graph is shared across all activity
        # modes so users can tweak ops while looking at the viz.
        vertical_splitter = QSplitter(Qt.Orientation.Vertical, self)
        vertical_splitter.addWidget(self._content_stack)
        vertical_splitter.addWidget(self._pipeline_view)
        vertical_splitter.setStretchFactor(0, 5)
        vertical_splitter.setStretchFactor(1, 1)
        vertical_splitter.setCollapsible(0, False)
        vertical_splitter.setCollapsible(1, False)
        vertical_splitter.setSizes([700, 200])
        self._pipeline_view.setMinimumHeight(80)
        self._vertical_splitter = vertical_splitter

        # Activity bar on the far left — collapsible mode switcher.
        self._activity_bar = ActivityBar(self)
        self._activity_bar.mode_changed.connect(self._on_activity_mode_changed)

        editor_page = QWidget(self)
        editor_layout = QHBoxLayout(editor_page)
        editor_layout.setContentsMargins(0, 0, 0, 0)
        editor_layout.setSpacing(0)
        editor_layout.addWidget(self._activity_bar)
        editor_layout.addWidget(vertical_splitter, 1)

        # Dataset gallery "page" — same window, different tab. Clicking a
        # thumbnail flips back to the Editor tab with that image loaded.
        self._dataset_page = DatasetPage(self)
        self._dataset_page.image_chosen.connect(self._on_dataset_image_chosen)

        self._tabs = QTabWidget(self)
        self._tabs.setDocumentMode(True)
        self._editor_tab_index = self._tabs.addTab(editor_page, "Editor")
        self._dataset_tab_index = self._tabs.addTab(self._dataset_page, "Dataset")
        self.setCentralWidget(self._tabs)

        self._video_controller = VideoFeedController(self)
        self._video_controller.frame_ready.connect(self._on_video_frame)
        self._video_controller.finished.connect(self._on_video_finished)
        self._recorder: VideoRecorder | None = None
        self._help_dialog: HelpDialog | None = None

        self.setStatusBar(QStatusBar(self))

        self._build_menu()
        self._tools_panel = ImageToolsPanel(
            split_action=self._split_action,
            downscale_action=self._downscale_action,
            select_roi_action=self._select_roi_action,
            clear_roi_action=self._clear_roi_action,
            randomize_paste_action=self._randomize_paste_action,
            clear_paste_action=self._clear_paste_action,
            parent=self,
        )
        self._install_tools_sidebar()
        self._install_action_bar()
        self._setup_worker()
        self._setup_debouncer()
        self._wire_signals()
        self._setup_ai_cache_persistence()
        self._restore_ui_state()

    def _settings(self) -> QSettings:
        # Reads QApplication.organizationName / applicationName, which
        # are set in `ui.app.run` before MainWindow is constructed.
        return QSettings()

    def _restore_ui_state(self) -> None:
        """Apply the persisted window geometry + splitter sizes +
        downscale toggle. Stored values that fail to apply (e.g. a
        splitter size for a layout that has changed shape since save)
        are ignored — we want a slightly off but functional layout
        rather than a launch crash on stale state."""
        settings = self._settings()
        geometry = settings.value("window/geometry")
        if isinstance(geometry, (bytes, bytearray)):
            self.restoreGeometry(geometry)
        state = settings.value("window/state")
        if isinstance(state, (bytes, bytearray)):
            self.restoreState(state)

        top_sizes = settings.value("splitter/top")
        if isinstance(top_sizes, list):
            try:
                self._top_splitter.setSizes([int(x) for x in top_sizes])
            except (TypeError, ValueError):
                pass
        vertical_sizes = settings.value("splitter/vertical")
        if isinstance(vertical_sizes, list):
            try:
                self._vertical_splitter.setSizes([int(x) for x in vertical_sizes])
            except (TypeError, ValueError):
                pass

        downscale = settings.value("view/downscale")
        if downscale is not None:
            enabled = _to_bool(downscale)
            self._downscale_action.setChecked(enabled)
            self._downscale_enabled = enabled

        mode = settings.value("activity/mode")
        if isinstance(mode, str) and mode:
            self._activity_bar.set_current_mode(mode)

    def _save_ui_state(self) -> None:
        settings = self._settings()
        settings.setValue("window/geometry", self.saveGeometry())
        settings.setValue("window/state", self.saveState())
        settings.setValue("splitter/top", self._top_splitter.sizes())
        settings.setValue("splitter/vertical", self._vertical_splitter.sizes())
        settings.setValue("view/downscale", self._downscale_enabled)
        settings.setValue("activity/mode", self._activity_bar.current_mode)

    def _setup_ai_cache_persistence(self) -> None:
        """Hydrate every AI backend's cache from disk on launch. A
        corrupt or missing file just leaves the caches empty — the user
        will repay the cost on their next inference and we'll write a
        fresh file on close."""
        self._ai_cache_path = default_cache_path()
        try:
            count = load_caches(self._ai_cache_path, ai_ops.all_backends())
            if count > 0:
                self.statusBar().showMessage(
                    f"Loaded {count} cached AI response(s) from disk"
                )
        except OSError:
            # Don't crash the app over an unreadable cache file.
            pass

    def _install_tools_sidebar(self) -> None:
        """Attach the ImageToolsPanel next to the image. Called after the View-
        menu actions exist so the panel can bind to them."""
        row_layout = self._image_row.layout()
        assert row_layout is not None
        row_layout.addWidget(self._tools_panel)

    def _install_action_bar(self) -> None:
        """Attach the ImageActionBar below the image row. Wires File-menu
        actions to the new toolbar so the menu remains the source of truth."""
        self._action_bar = ImageActionBar(
            open_image_action=self._open_image_action,
            open_dataset_action=self._open_dataset_action,
            open_camera_action=self._open_camera_action,
            open_video_action=self._open_video_action,
            save_image_action=self._save_image_action,
            record_action=self._record_action,
            stop_recording_action=self._stop_recording_action,
            stop_capture_action=self._stop_capture_action,
            pause_capture_action=self._pause_capture_action,
            parent=self._image_with_tools,
        )
        layout = self._image_with_tools.layout()
        assert layout is not None
        layout.addWidget(self._action_bar)

    def _build_menu(self) -> None:
        self._build_file_menu()
        self._build_view_menu()
        self._build_tools_menu()
        self._build_help_menu()

    def _build_tools_menu(self) -> None:
        tools_menu = self.menuBar().addMenu("&Tools")
        clear_ai_action = QAction("Clear &AI cache", self)
        clear_ai_action.setToolTip(
            "Drop every cached VLM / CLIP / OWL-ViT / BLIP-2 response so the "
            "next pipeline run hits the model again."
        )
        clear_ai_action.triggered.connect(self._on_clear_ai_cache)
        tools_menu.addAction(clear_ai_action)

    def _on_clear_ai_cache(self) -> None:
        ai_ops.clear_cache()
        # Persist the now-empty cache so a restart doesn't reload stale
        # entries from disk.
        try:
            save_caches(self._ai_cache_path, ai_ops.all_backends())
        except OSError as exc:
            self.statusBar().showMessage(f"Could not write AI cache: {exc}")
            return
        self.statusBar().showMessage("AI cache cleared")
        # Force a re-render so any "AI Response" panel showing a stale
        # cached reply refreshes against the now-empty store.
        self._request_preview()

    def _build_help_menu(self) -> None:
        help_menu = self.menuBar().addMenu("&Help")
        guide_action = QAction("&Operation Guide…", self)
        guide_action.setShortcut("F1")
        guide_action.triggered.connect(self._on_show_help)
        help_menu.addAction(guide_action)

    def _build_file_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")

        self._open_image_action = QAction("&Open Image…", self)
        self._open_image_action.setShortcut(QKeySequence.StandardKey.Open)
        self._open_image_action.triggered.connect(self._on_open)
        file_menu.addAction(self._open_image_action)

        self._open_dataset_action = QAction("Open &Dataset…", self)
        self._open_dataset_action.setShortcut("Ctrl+D")
        self._open_dataset_action.triggered.connect(self._on_open_dataset)
        file_menu.addAction(self._open_dataset_action)

        self._open_camera_action = QAction("Open &Camera", self)
        self._open_camera_action.triggered.connect(self._on_open_camera)
        file_menu.addAction(self._open_camera_action)

        self._open_video_action = QAction("Open &Video…", self)
        self._open_video_action.triggered.connect(self._on_open_video)
        file_menu.addAction(self._open_video_action)

        self._pause_capture_action = QAction("&Pause Video", self)
        self._pause_capture_action.setShortcut("Space")
        self._pause_capture_action.triggered.connect(self._on_toggle_pause)
        self._pause_capture_action.setEnabled(False)
        file_menu.addAction(self._pause_capture_action)

        self._stop_capture_action = QAction("&Stop Capture", self)
        self._stop_capture_action.triggered.connect(self._on_stop_capture)
        self._stop_capture_action.setEnabled(False)
        file_menu.addAction(self._stop_capture_action)

        file_menu.addSeparator()

        self._save_image_action = QAction("Save Processed &Image…", self)
        self._save_image_action.setShortcut("Ctrl+S")
        self._save_image_action.triggered.connect(self._on_save_image)
        file_menu.addAction(self._save_image_action)

        self._record_action = QAction("&Record Video…", self)
        self._record_action.triggered.connect(self._on_record_video)
        file_menu.addAction(self._record_action)

        self._stop_recording_action = QAction("S&top Recording", self)
        self._stop_recording_action.triggered.connect(self._on_stop_recording)
        self._stop_recording_action.setEnabled(False)
        file_menu.addAction(self._stop_recording_action)

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

        batch_action = QAction("&Bulk Export Dataset…", self)
        batch_action.triggered.connect(self._on_batch_process)
        file_menu.addAction(batch_action)

        file_menu.addSeparator()

        quit_action = QAction("&Quit", self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

    def _build_view_menu(self) -> None:
        view_menu = self.menuBar().addMenu("&View")

        self._split_action = QAction("&Before/After split", self)
        self._split_action.setCheckable(True)
        self._split_action.setShortcut("Ctrl+B")
        self._split_action.toggled.connect(self._image_view.set_split_enabled)
        view_menu.addAction(self._split_action)

        view_menu.addSeparator()

        self._downscale_action = QAction(
            f"&Downscale large previews (>{PREVIEW_MAX_DIM} px)", self
        )
        self._downscale_action.setCheckable(True)
        self._downscale_action.setChecked(self._downscale_enabled)
        self._downscale_action.toggled.connect(self._on_downscale_toggled)
        view_menu.addAction(self._downscale_action)

        view_menu.addSeparator()

        self._select_roi_action = QAction("Select &ROI", self)
        self._select_roi_action.setCheckable(True)
        self._select_roi_action.setShortcut("Ctrl+R")
        self._select_roi_action.toggled.connect(self._image_view.set_roi_mode)
        view_menu.addAction(self._select_roi_action)

        self._clear_roi_action = QAction("Clea&r ROI", self)
        self._clear_roi_action.triggered.connect(self._on_clear_roi)
        view_menu.addAction(self._clear_roi_action)

        self._randomize_paste_action = QAction("Randomize &paste destination", self)
        self._randomize_paste_action.setShortcut("Ctrl+Shift+R")
        self._randomize_paste_action.triggered.connect(self._on_randomize_paste)
        view_menu.addAction(self._randomize_paste_action)

        self._clear_paste_action = QAction("Clear &paste destination", self)
        self._clear_paste_action.triggered.connect(self._on_clear_paste)
        view_menu.addAction(self._clear_paste_action)

        view_menu.addSeparator()

        # Activity bar shortcuts — Ctrl+1/2/3 switches Op/2D/3D mode.
        # The actions are owned by the menu so the shortcuts work even
        # when the activity bar isn't focused.
        op_action = QAction("&Operations editor\tCtrl+1", self)
        op_action.setShortcut("Ctrl+1")
        op_action.triggered.connect(
            lambda: self._activity_bar.set_current_mode(ActivityBar.MODE_OP)
        )
        view_menu.addAction(op_action)

        viz2d_action = QAction("2&D visualization\tCtrl+2", self)
        viz2d_action.setShortcut("Ctrl+2")
        viz2d_action.triggered.connect(
            lambda: self._activity_bar.set_current_mode(ActivityBar.MODE_2D)
        )
        view_menu.addAction(viz2d_action)

        viz3d_action = QAction("&3D visualization\tCtrl+3", self)
        viz3d_action.setShortcut("Ctrl+3")
        viz3d_action.triggered.connect(
            lambda: self._activity_bar.set_current_mode(ActivityBar.MODE_3D)
        )
        view_menu.addAction(viz3d_action)

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
        self._param_panel.run_requested.connect(self._on_run_requested)
        self._image_view.roi_changed.connect(self._on_roi_drawn)
        self._image_view.paste_destination_changed.connect(self._on_paste_destination_dragged)
        # VLM streams emit `progress` from worker threads as tokens arrive;
        # we route that through the existing debounced preview path so the
        # banner refreshes mid-generation without bypassing throttling.
        streaming_bus().progress.connect(self._request_preview)

    def _on_run_requested(self, node_id: str) -> None:
        """Authorize a manual-trigger node (currently only the VLM op)
        to spawn its backend call on the next pipeline run."""
        from cvstudio.operations import ai as ai_op

        ai_op.authorize_node(node_id)

    def _on_open(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            str(Path.home()),
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)",
        )
        if not path:
            return
        image = read_image(path)
        if image is None:
            QMessageBox.warning(self, "Open failed", f"Could not read image: {path}")
            return
        self._stop_capture_if_active()
        self._source_image = image
        self._refresh_preview_source(path_for_status=path)

    def _on_open_camera(self) -> None:
        self._start_capture(0, label="Camera")

    def _on_open_video(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(
            self,
            "Open Video",
            str(Path.home()),
            "Videos (*.mp4 *.mov *.avi *.mkv *.webm)",
        )
        if not path:
            return
        self._start_capture(path, label=path)

    def _on_stop_capture(self) -> None:
        self._stop_capture_if_active()
        self.statusBar().showMessage("Capture stopped")

    def _on_toggle_pause(self) -> None:
        if not self._video_controller.is_active():
            return
        if self._video_controller.is_paused():
            self._video_controller.resume()
            self._pause_capture_action.setText("&Pause Video")
            self.statusBar().showMessage("Resumed")
        else:
            self._video_controller.pause()
            self._pause_capture_action.setText("&Resume Video")
            self.statusBar().showMessage("Paused")

    def _start_capture(self, source: int | str, *, label: str) -> None:
        try:
            video_source = VideoSource(source)
            self._video_controller.start(video_source)
        except RuntimeError as exc:
            QMessageBox.warning(self, "Capture failed", str(exc))
            return
        self._stop_capture_action.setEnabled(True)
        self._pause_capture_action.setEnabled(True)
        self._pause_capture_action.setText("&Pause Video")
        self.statusBar().showMessage(f"Streaming from {label}")

    def _stop_capture_if_active(self) -> None:
        if self._video_controller.is_active():
            self._video_controller.stop()
        self._stop_capture_action.setEnabled(False)
        self._pause_capture_action.setEnabled(False)
        self._pause_capture_action.setText("&Pause Video")

    def _on_video_frame(self, frame: object) -> None:
        if not isinstance(frame, np.ndarray):
            self._video_controller.mark_processed()
            return
        self._source_image = frame
        # Mirror the static-image path but skip the debounce — the controller
        # already gates frame arrivals on the worker being free.
        self._preview_source = (
            downscale_for_preview(self._source_image)
            if self._downscale_enabled
            else self._source_image
        )
        self._image_view.set_before(self._preview_source)
        self._image_view.set_roi(self._pipeline_roi_in_preview_coords())
        self._image_view.set_paste_rect(self._pipeline_paste_rect_in_preview_coords())
        self._dispatch_preview()

    def _on_video_finished(self) -> None:
        self._stop_capture_action.setEnabled(False)
        self._pause_capture_action.setEnabled(False)
        self._pause_capture_action.setText("&Pause Video")
        if self._recorder is not None:
            self._on_stop_recording()
        self.statusBar().showMessage("Capture finished")

    def _refresh_preview_source(self, path_for_status: str | None = None) -> None:
        """Rebuild `_preview_source` from `_source_image`, applying downscale
        when enabled. Re-syncs the ROI overlay onto the new preview coordinate
        space and emits a fresh preview request."""
        if self._source_image is None:
            self._preview_source = None
            self._image_view.set_before(None)
            self._image_view.set_roi(None)
            self._request_preview()
            return
        self._preview_source = (
            downscale_for_preview(self._source_image)
            if self._downscale_enabled
            else self._source_image
        )
        self._image_view.set_before(self._preview_source)
        # ROI overlay lives in scene coords (== preview coords). If the preview
        # source changed scale, the overlay needs to be re-projected from the
        # canonical full-source coordinates stored on the pipeline.
        self._image_view.set_roi(self._pipeline_roi_in_preview_coords())
        self._image_view.set_paste_rect(self._pipeline_paste_rect_in_preview_coords())
        if path_for_status is not None:
            self.statusBar().showMessage(self._format_status(path_for_status))
        self._request_preview()

    def _preview_scale(self) -> tuple[float, float]:
        """(sx, sy) factors converting source-coords to preview-coords —
        i.e. `preview = source * sx`. Returns (1, 1) when there is no
        downscaling (preview source == full source)."""
        if self._source_image is None or self._preview_source is None:
            return 1.0, 1.0
        if self._preview_source is self._source_image:
            return 1.0, 1.0
        sh, sw = self._source_image.shape[:2]
        ph, pw = self._preview_source.shape[:2]
        return pw / sw, ph / sh

    def _pipeline_roi_in_preview_coords(self) -> tuple[int, int, int, int] | None:
        if self._pipeline.roi is None:
            return None
        sx, sy = self._preview_scale()
        roi = self._pipeline.roi
        return (
            round(roi.x * sx),
            round(roi.y * sy),
            round(roi.width * sx),
            round(roi.height * sy),
        )

    def _pipeline_paste_in_preview_coords(self) -> tuple[int, int] | None:
        if self._pipeline.roi_paste_to is None:
            return None
        sx, sy = self._preview_scale()
        px, py = self._pipeline.roi_paste_to
        return round(px * sx), round(py * sy)

    def _pipeline_paste_rect_in_preview_coords(self) -> tuple[int, int, int, int] | None:
        """Return the (x, y, w, h) of the paste destination in preview coords,
        suitable for the cyan overlay. None if no paste-to is set."""
        if self._pipeline.roi is None or self._pipeline.roi_paste_to is None:
            return None
        roi_preview = self._pipeline_roi_in_preview_coords()
        paste_xy = self._pipeline_paste_in_preview_coords()
        if roi_preview is None or paste_xy is None:
            return None
        _x, _y, w, h = roi_preview
        px, py = paste_xy
        return px, py, w, h

    def _format_status(self, path: str) -> str:
        assert self._source_image is not None
        sh, sw = self._source_image.shape[:2]
        if self._preview_source is None or self._preview_source is self._source_image:
            return f"Loaded {path}  ·  {sw}x{sh}"
        ph, pw = self._preview_source.shape[:2]
        return f"Loaded {path}  ·  {sw}x{sh}  ·  preview at {pw}x{ph}"

    def _on_downscale_toggled(self, enabled: bool) -> None:
        self._downscale_enabled = enabled
        self._refresh_preview_source()

    def _on_roi_drawn(self, x: int, y: int, w: int, h: int) -> None:
        """User finished drawing a rectangle in ROI mode. Incoming coords are
        in preview-source space; translate to full-source space before storing
        so code export and pipeline.execute stay correct under downscaling."""
        sx, sy = self._preview_scale()
        if sx <= 0 or sy <= 0:
            return
        fx = round(x / sx)
        fy = round(y / sy)
        fw = max(1, round(w / sx))
        fh = max(1, round(h / sy))
        self._pipeline.roi = Roi(x=fx, y=fy, width=fw, height=fh)
        # Re-sync the visual overlay from the canonical pipeline.roi so any
        # rounding stays consistent between preview and saved state.
        self._image_view.set_roi(self._pipeline_roi_in_preview_coords())
        self.statusBar().showMessage(f"ROI set: {fw}x{fh} at ({fx}, {fy})")
        # Exit ROI selection mode automatically — the user just placed one.
        self._select_roi_action.setChecked(False)
        self._request_preview()

    def _on_clear_roi(self) -> None:
        self._pipeline.roi = None
        self._pipeline.roi_paste_to = None
        self._image_view.set_roi(None)
        self._image_view.set_paste_rect(None)
        self._select_roi_action.setChecked(False)
        self.statusBar().showMessage("ROI cleared")
        self._request_preview()

    def _on_randomize_paste(self) -> None:
        roi = self._pipeline.roi
        if roi is None or self._source_image is None:
            self.statusBar().showMessage("Select an ROI first")
            return
        img_h, img_w = self._source_image.shape[:2]
        max_x = max(0, img_w - roi.width)
        max_y = max(0, img_h - roi.height)
        new_x = random.randint(0, max_x) if max_x > 0 else 0
        new_y = random.randint(0, max_y) if max_y > 0 else 0
        self._pipeline.roi_paste_to = (new_x, new_y)
        self._image_view.set_paste_rect(self._pipeline_paste_rect_in_preview_coords())
        self.statusBar().showMessage(f"Paste destination set to ({new_x}, {new_y})")
        self._request_preview()

    def _on_paste_destination_dragged(self, x: int, y: int) -> None:
        """User dragged inside the green ROI to reposition the cyan paste
        destination. Coords are in preview-source space; translate to full
        source coords before storing."""
        if self._pipeline.roi is None:
            return
        sx, sy = self._preview_scale()
        if sx <= 0 or sy <= 0:
            return
        fx = round(x / sx)
        fy = round(y / sy)
        self._pipeline.roi_paste_to = (fx, fy)
        # The image-view's overlay is already at the right place (its drag
        # handler set it eagerly); we just need to re-dispatch the preview.
        self.statusBar().showMessage(f"Paste destination: ({fx}, {fy})")
        self._request_preview()

    def _on_clear_paste(self) -> None:
        if self._pipeline.roi_paste_to is None:
            return
        self._pipeline.roi_paste_to = None
        self._image_view.set_paste_rect(None)
        self.statusBar().showMessage("Paste destination cleared")
        self._request_preview()

    def _on_new_pipeline(self) -> None:
        self._pipeline.clear()
        self._pipeline_view.refresh()
        self._param_panel.set_node(None)
        self._image_view.set_roi(None)
        self._image_view.set_paste_rect(None)
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
        # Sync the visual ROI overlay with whatever the loaded pipeline carries.
        self._image_view.set_roi(self._pipeline_roi_in_preview_coords())
        self._image_view.set_paste_rect(self._pipeline_paste_rect_in_preview_coords())
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

    def _on_batch_process(self) -> None:
        dialog = BatchDialog(self._pipeline.execute, parent=self)
        dialog.exec()

    def _on_open_dataset(self) -> None:
        """Switch to the in-window Dataset tab. If no folder has been picked
        yet, prompt for one immediately so the user lands somewhere
        actionable instead of an empty grid."""
        self._tabs.setCurrentIndex(self._dataset_tab_index)
        if not self._dataset_page.has_folder():
            self._dataset_page.prompt_for_folder()

    def _on_dataset_image_chosen(self, path: str) -> None:
        """User clicked a thumbnail — load that file as the new source and
        snap the tab back to the editor so they can immediately tune the
        pipeline and Save Image."""
        image = read_image(path)
        if image is None:
            QMessageBox.warning(self, "Open failed", f"Could not read image: {path}")
            return
        self._stop_capture_if_active()
        self._source_image = image
        self._refresh_preview_source(path_for_status=path)
        self._dataset_page.mark_active(path)
        self._tabs.setCurrentIndex(self._editor_tab_index)

    def _on_show_help(self) -> None:
        """Open (or focus) the non-modal Operation Guide window."""
        from cvstudio.core.pipeline import SOURCE_SPEC
        from cvstudio.operations import all_builtin_specs

        if self._help_dialog is None:
            specs = (SOURCE_SPEC, *all_builtin_specs())
            self._help_dialog = HelpDialog(specs, parent=self)
        self._help_dialog.show()
        self._help_dialog.raise_()
        self._help_dialog.activateWindow()

    def _on_save_image(self) -> None:
        """Run the pipeline on the FULL-resolution source and write the result
        to disk. Bypasses the downscaled preview path so the saved file keeps
        whatever the user actually loaded."""
        if self._source_image is None:
            QMessageBox.information(
                self, "No image", "Load an image, camera, or video first."
            )
            return
        path, _filter = QFileDialog.getSaveFileName(
            self,
            "Save processed image",
            str(Path.home() / "processed.png"),
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)",
        )
        if not path:
            return
        try:
            result = self._pipeline.execute(self._source_image)
            if not cv2.imwrite(path, result):
                raise OSError(f"cv2.imwrite returned False for {path}")
        except (OSError, ValueError, TypeError) as exc:
            QMessageBox.warning(self, "Save failed", str(exc))
            return
        self.statusBar().showMessage(f"Saved processed image to {path}")

    def _on_record_video(self) -> None:
        if not self._video_controller.is_active():
            QMessageBox.information(
                self,
                "No capture",
                "Open a camera or video first, then start recording.",
            )
            return
        if self._recorder is not None:
            return  # already recording
        path, _filter = QFileDialog.getSaveFileName(
            self,
            "Record processed video",
            str(Path.home() / "recording.mp4"),
            "Videos (*.mp4 *.avi)",
        )
        if not path:
            return
        source = self._video_controller.current_source()
        fps = source.fps() if source else 0.0
        if fps <= 0:
            fps = 30.0
        fourcc = "mp4v" if path.lower().endswith(".mp4") else "MJPG"
        try:
            self._recorder = VideoRecorder(path, fps=fps, fourcc=fourcc)
        except OSError as exc:
            QMessageBox.warning(self, "Record failed", str(exc))
            return
        self._stop_recording_action.setEnabled(True)
        self._record_action.setEnabled(False)
        self.statusBar().showMessage(f"Recording to {path}")

    def _on_stop_recording(self) -> None:
        if self._recorder is not None:
            self._recorder.close()
            self._recorder = None
        self._stop_recording_action.setEnabled(False)
        self._record_action.setEnabled(True)
        self.statusBar().showMessage("Recording stopped")

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

    def _on_activity_mode_changed(self, mode: str) -> None:
        """Swap the central content stack and re-host the shared ParameterPanel.

        The ParameterPanel is a single widget reused across all four modes.
        We reparent it on every mode switch so the user always sees one
        consistent param view inside the current page.

        Op and AI both use the standard 3-column page — AI just filters
        the left catalog down to the "AI" category so the user does not
        have to hunt through every OpenCV op to find the model nodes.
        """
        if mode == ActivityBar.MODE_2D:
            self._content_stack.setCurrentWidget(self._viz_2d_page)
            self._viz_2d_page.install_param_panel(self._param_panel)
            self._catalog.set_category_filter(None)
        elif mode == ActivityBar.MODE_3D:
            self._content_stack.setCurrentWidget(self._viz_3d_page)
            self._viz_3d_page.install_param_panel(self._param_panel)
            self._catalog.set_category_filter(None)
        else:
            self._content_stack.setCurrentWidget(self._op_page)
            # Hand the panel back to its Op-mode home — the splitter that
            # historically hosted it.
            self._param_panel.setParent(self._param_panel_op_layout)
            self._param_panel_op_layout.insertWidget(0, self._param_panel)
            self._param_panel.show()
            if mode == ActivityBar.MODE_AI:
                self._catalog.set_category_filter("AI")
            else:
                self._catalog.set_category_filter(None)
        # Re-bind so the freshly-reparented widget rebuilds its form
        # against the currently-selected node (avoids stale state).
        self._on_selection_changed(self._pipeline_view._selected_index)

    def _request_preview(self) -> None:
        self._debounce.start()  # restarts if already running

    def _dispatch_preview(self) -> None:
        if self._preview_source is None:
            self._image_view.set_image(None)
            self._histogram_panel.clear()
            self._viz_2d_page.clear()
            self._viz_3d_page.clear()
            self._pipeline_view.clear_timings()
            return
        steps = tuple(
            (node.spec.func, dict(node.params), node.id)
            for node in self._pipeline.nodes
            if node.enabled
        )
        self._next_request_id += 1
        self._latest_request_id = self._next_request_id
        request = PipelineRequest(
            request_id=self._next_request_id,
            image=self._preview_source,
            steps=steps,
            roi=self._pipeline_roi_in_preview_coords(),
            roi_paste_to=self._pipeline_paste_in_preview_coords(),
        )
        self._execute_requested.emit(request)

    def _on_worker_result(self, request_id: int, image: object, timings: object) -> None:
        if request_id != self._latest_request_id:
            return  # a newer request has already superseded this one
        if not isinstance(image, np.ndarray):
            return
        self._image_view.set_image(image)
        self._histogram_panel.set_image(image)
        self._viz_2d_page.set_image(image)
        self._viz_3d_page.set_image(image)
        self._apply_timings(timings)
        if self._recorder is not None and self._video_controller.is_active():
            try:
                self._recorder.write(image)
            except (OSError, ValueError, TypeError) as exc:
                QMessageBox.warning(self, "Recording error", str(exc))
                self._on_stop_recording()
        if self._video_controller.is_active():
            # Worker is free again — let the controller pull the next frame.
            self._video_controller.mark_processed()

    def _apply_timings(self, timings: object) -> None:
        """Map an enabled-only timings tuple back to per-pipeline-node timings."""
        if not isinstance(timings, tuple):
            return
        per_node: list[float | None] = []
        iterator = iter(timings)
        for node in self._pipeline.nodes:
            if node.enabled:
                per_node.append(next(iterator, None))
            else:
                per_node.append(None)
        self._pipeline_view.set_timings(per_node)

    def _on_worker_failed(self, request_id: int, message: str) -> None:
        if request_id != self._latest_request_id:
            return
        self.statusBar().showMessage(f"Pipeline error: {message}")

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 (Qt override)
        if self._recorder is not None:
            self._recorder.close()
            self._recorder = None
        self._stop_capture_if_active()
        try:
            save_caches(self._ai_cache_path, ai_ops.all_backends())
        except OSError:
            # An unwritable cache file mustn't block shutdown.
            pass
        try:
            self._save_ui_state()
        except Exception:  # noqa: BLE001 — shutdown path, never raise
            pass
        self._worker_thread.quit()
        self._worker_thread.wait(2000)
        super().closeEvent(event)
