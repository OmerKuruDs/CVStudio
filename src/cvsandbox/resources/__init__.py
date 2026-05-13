"""Bundled static assets (icon, etc.). Loaded at runtime via __file__-relative paths."""

from __future__ import annotations

from pathlib import Path

_RESOURCES_DIR = Path(__file__).resolve().parent
ICON_PATH = _RESOURCES_DIR / "icon.svg"

__all__ = ["ICON_PATH"]
