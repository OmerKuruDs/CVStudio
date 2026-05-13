"""QApplication bootstrap.

`run()` is the single entry point used by `cvsandbox.__main__`. It is split out
from MainWindow construction so tests can build the window without spinning the
event loop.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from cvsandbox.operations import load_builtin_operations
from cvsandbox.resources import ICON_PATH
from cvsandbox.ui.main_window import MainWindow


def run(argv: Sequence[str] | None = None) -> int:
    load_builtin_operations()
    _apply_windows_app_id()
    app = QApplication(list(argv) if argv is not None else sys.argv)
    app.setApplicationName("cvsandbox")
    app.setWindowIcon(QIcon(str(ICON_PATH)))
    window = MainWindow()
    window.show()
    return app.exec()


def _apply_windows_app_id() -> None:
    """Tag the process with an explicit AppUserModelID so Windows uses our icon
    in the taskbar instead of inheriting the python.exe icon."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("cvsandbox.app")
    except (OSError, AttributeError):
        pass
