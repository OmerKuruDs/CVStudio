from __future__ import annotations

from cvsandbox.resources import ICON_PATH


def test_icon_is_bundled_and_non_empty() -> None:
    assert ICON_PATH.exists(), f"icon.svg missing at {ICON_PATH}"
    contents = ICON_PATH.read_bytes()
    assert contents.startswith((b"<svg", b"<?xml"))
    assert len(contents) > 100  # not an empty stub


def test_icon_path_is_under_cvsandbox_package() -> None:
    # Sanity: resource resolves to a path inside the installed/source package.
    assert ICON_PATH.parent.name == "resources"
    assert ICON_PATH.parent.parent.name == "cvsandbox"
