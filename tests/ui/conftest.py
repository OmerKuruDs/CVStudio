"""Shared fixtures for UI tests.

PySide6 widgets need a QApplication; we create one offscreen so tests run
headless in CI. The QApplication lives for the whole test session — Qt only
allows one per process.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="session")
def qapp() -> Iterator[QApplication]:
    app = QApplication.instance() or QApplication([])
    yield app  # type: ignore[misc]
