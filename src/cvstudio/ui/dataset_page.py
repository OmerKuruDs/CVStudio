"""DatasetPage — embedded gallery view of a folder of images.

Lives inside MainWindow's QTabWidget as a second page next to the editor.
Clicking a thumbnail emits `image_chosen(path)`; MainWindow flips back to
the Editor tab with that image loaded as the current source.

Thumbnails are loaded synchronously with cv2's reduced-decode path, which
is fast enough for typical folders without needing a worker thread. Large
datasets get a "Loading…" status so the user knows the page is busy.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListView,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from cvstudio.core.batch import discover_images
from cvstudio.core.image_io import read_thumbnail
from cvstudio.ui.image_view import _ndarray_to_qpixmap

THUMB_PX = 160


class DatasetPage(QWidget):
    image_chosen = Signal(str)
    """Absolute path of the image the user clicked on."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._folder: Path | None = None
        self._highlighted_path: str | None = None

        path_row = QHBoxLayout()
        path_row.addWidget(QLabel("Folder:"))
        self._folder_edit = QLineEdit()
        self._folder_edit.setReadOnly(True)
        self._folder_edit.setPlaceholderText("Pick a folder of images to begin")
        pick_button = QPushButton("Choose folder…")
        pick_button.clicked.connect(self._on_pick_folder)
        path_row.addWidget(self._folder_edit, 1)
        path_row.addWidget(pick_button)

        self._grid = QListWidget(self)
        self._grid.setViewMode(QListView.ViewMode.IconMode)
        self._grid.setIconSize(QSize(THUMB_PX, THUMB_PX))
        self._grid.setGridSize(QSize(THUMB_PX + 24, THUMB_PX + 40))
        self._grid.setResizeMode(QListView.ResizeMode.Adjust)
        self._grid.setMovement(QListView.Movement.Static)
        self._grid.setUniformItemSizes(True)
        self._grid.setSpacing(6)
        self._grid.setWordWrap(True)
        self._grid.itemClicked.connect(self._on_pick)

        self._status = QLabel("No folder selected yet.")
        self._status.setStyleSheet("color: #475569;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.addLayout(path_row)
        layout.addWidget(self._grid, 1)
        layout.addWidget(self._status)

    # ------------------------------------------------------------------ public

    def load_folder(self, folder: Path) -> None:
        self._folder = folder
        self._folder_edit.setText(str(folder))
        self._grid.clear()
        self._highlighted_path = None
        paths = discover_images(folder)
        if not paths:
            self._status.setText("No supported images in this folder.")
            return
        self._status.setText(f"Loading {len(paths)} thumbnails…")
        QApplication.processEvents()
        for path in paths:
            self._append_item(path)
        self._status.setText(f"{len(paths)} image(s). Click one to load it.")

    def prompt_for_folder(self) -> None:
        """Open the folder picker if no folder has been loaded yet — used when
        the page is shown for the first time so the user lands somewhere
        actionable instead of an empty grid."""
        if self._folder is None:
            self._on_pick_folder()

    def mark_active(self, path: str | None) -> None:
        """Highlight the row whose payload matches `path` (or clear when
        None). Used by MainWindow to keep the gallery in sync with the
        currently-loaded source image."""
        self._highlighted_path = path
        for i in range(self._grid.count()):
            item = self._grid.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == path:
                self._grid.setCurrentRow(i)
                self._grid.scrollToItem(item)
                return

    def has_folder(self) -> bool:
        return self._folder is not None

    # ------------------------------------------------------------------ events

    def _on_pick_folder(self) -> None:
        start = self._folder_edit.text().strip() or str(Path.home())
        path = QFileDialog.getExistingDirectory(self, "Pick dataset folder", start)
        if path:
            self.load_folder(Path(path))

    def _on_pick(self, item: QListWidgetItem) -> None:
        payload = item.data(Qt.ItemDataRole.UserRole)
        if isinstance(payload, str):
            self.image_chosen.emit(payload)

    # ------------------------------------------------------------------ helpers

    def _append_item(self, path: Path) -> None:
        thumb = read_thumbnail(path, max_dim=THUMB_PX)
        if thumb is not None:
            pixmap = _ndarray_to_qpixmap(thumb)
            icon = QIcon(pixmap)
        else:
            placeholder = QPixmap(THUMB_PX, THUMB_PX)
            placeholder.fill(Qt.GlobalColor.lightGray)
            icon = QIcon(placeholder)
        item = QListWidgetItem(icon, path.name)
        item.setData(Qt.ItemDataRole.UserRole, str(path))
        item.setToolTip(str(path))
        self._grid.addItem(item)
