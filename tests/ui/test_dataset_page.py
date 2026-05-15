from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from cvstudio.ui.dataset_page import DatasetPage


def _write_image(path: Path, value: int = 50) -> None:
    img = np.full((40, 40, 3), value, dtype=np.uint8)
    cv2.imwrite(str(path), img)


@pytest.fixture
def dataset_folder(tmp_path: Path) -> Path:
    folder = tmp_path / "dataset"
    folder.mkdir()
    for name, value in [("a.png", 20), ("b.png", 80), ("c.png", 200)]:
        _write_image(folder / name, value)
    (folder / "notes.txt").write_text("not an image", encoding="utf-8")
    return folder


def test_dialog_loads_each_supported_image_as_a_grid_item(
    qapp: QApplication, dataset_folder: Path
) -> None:
    dialog = DatasetPage()
    dialog.load_folder(dataset_folder)
    qapp.processEvents()
    titles = [dialog._grid.item(i).text() for i in range(dialog._grid.count())]
    assert sorted(titles) == ["a.png", "b.png", "c.png"]
    dialog.deleteLater()


def test_dialog_ignores_non_image_files(qapp: QApplication, dataset_folder: Path) -> None:
    dialog = DatasetPage()
    dialog.load_folder(dataset_folder)
    qapp.processEvents()
    titles = [dialog._grid.item(i).text() for i in range(dialog._grid.count())]
    assert "notes.txt" not in titles
    dialog.deleteLater()


def test_clicking_thumbnail_emits_path(
    qapp: QApplication, dataset_folder: Path
) -> None:
    dialog = DatasetPage()
    dialog.load_folder(dataset_folder)
    qapp.processEvents()

    received: list[str] = []
    dialog.image_chosen.connect(received.append)
    dialog._on_pick(dialog._grid.item(0))
    qapp.processEvents()

    assert len(received) == 1
    assert received[0].endswith(("a.png", "b.png", "c.png"))
    dialog.deleteLater()


def test_mark_active_highlights_matching_thumbnail(
    qapp: QApplication, dataset_folder: Path
) -> None:
    dialog = DatasetPage()
    dialog.load_folder(dataset_folder)
    qapp.processEvents()

    target = str(dataset_folder / "b.png")
    dialog.mark_active(target)
    current = dialog._grid.currentItem()
    assert current is not None
    assert current.data(Qt.ItemDataRole.UserRole) == target
    dialog.deleteLater()


def test_empty_folder_shows_status_message(qapp: QApplication, tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    dialog = DatasetPage()
    dialog.load_folder(empty_dir)
    qapp.processEvents()
    assert dialog._grid.count() == 0
    assert "No supported images" in dialog._status.text()
    dialog.deleteLater()


def test_page_is_an_embeddable_qwidget(qapp: QApplication) -> None:
    """DatasetPage must be a plain QWidget so MainWindow can drop it into a
    QTabWidget — not a top-level dialog."""
    from PySide6.QtWidgets import QDialog, QWidget

    page = DatasetPage()
    assert isinstance(page, QWidget)
    assert not isinstance(page, QDialog)
    page.deleteLater()
