"""QApplication bootstrap.

`run()` is the single entry point used by `cvsandbox.__main__`. It is split out
from MainWindow construction so tests can build the window without spinning the
event loop.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from PySide6.QtWidgets import QApplication

from cvsandbox.operations import load_builtin_operations
from cvsandbox.ui.main_window import MainWindow


def run(argv: Sequence[str] | None = None) -> int:
    load_builtin_operations()
    app = QApplication(list(argv) if argv is not None else sys.argv)
    app.setApplicationName("cvsandbox")
    window = MainWindow()
    window.show()
    return app.exec()
