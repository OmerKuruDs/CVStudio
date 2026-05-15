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
from cvsandbox.resources import ARROW_DOWN_PATH, ARROW_UP_PATH, ICON_PATH, THEME_QSS_PATH
from cvsandbox.ui.main_window import MainWindow


def run(argv: Sequence[str] | None = None) -> int:
    load_builtin_operations()
    _apply_windows_app_id()
    app = QApplication(list(argv) if argv is not None else sys.argv)
    # Org + app names feed both QStandardPaths (used by the AI cache
    # storage) and QSettings (window geometry / splitter sizes). Set
    # them BEFORE creating MainWindow so the first cache load uses the
    # right per-OS directory.
    app.setOrganizationName("cvsandbox")
    app.setOrganizationDomain("cvsandbox.local")
    app.setApplicationName("cvsandbox")
    app.setWindowIcon(QIcon(str(ICON_PATH)))
    if THEME_QSS_PATH.exists():
        qss = THEME_QSS_PATH.read_text(encoding="utf-8")
        # QSS `url(...)` resolves against the CWD without help — substitute
        # bundled-resource placeholders with absolute forward-slash paths so
        # spinbox arrows etc. render regardless of where the app is launched.
        qss = qss.replace("__ARROW_UP__", ARROW_UP_PATH.as_posix())
        qss = qss.replace("__ARROW_DOWN__", ARROW_DOWN_PATH.as_posix())
        app.setStyleSheet(qss)
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
