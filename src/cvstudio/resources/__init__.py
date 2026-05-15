"""Bundled static assets (icon, etc.). Loaded at runtime via __file__-relative paths."""

from __future__ import annotations

from pathlib import Path

_RESOURCES_DIR = Path(__file__).resolve().parent
ICON_PATH = _RESOURCES_DIR / "icon.svg"
THEME_QSS_PATH = _RESOURCES_DIR / "theme.qss"
ARROW_UP_PATH = _RESOURCES_DIR / "arrow_up.svg"
ARROW_DOWN_PATH = _RESOURCES_DIR / "arrow_down.svg"

__all__ = ["ARROW_DOWN_PATH", "ARROW_UP_PATH", "ICON_PATH", "THEME_QSS_PATH"]
