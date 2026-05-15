from __future__ import annotations

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from cvstudio.core.operation import OperationSpec, Parameter
from cvstudio.ui.help_dialog import HelpDialog


def _spec(spec_id: str, name: str, category: str) -> OperationSpec:
    return OperationSpec(
        id=spec_id,
        name=name,
        category=category,
        description="",
        parameters=(Parameter(name="x", kind="int", default=0, min=0, max=10),),
        func=lambda image, x: image,
    )


@pytest.fixture
def specs() -> list[OperationSpec]:
    return [
        _spec("filtering.gaussian_blur", "Gaussian Blur", "Filtering"),
        _spec("threshold.binary", "Binary Threshold", "Threshold"),
        _spec("source.image", "Source", "Source"),
    ]


def test_help_dialog_lists_every_spec_plus_category_headers(
    qapp: QApplication, specs: list[OperationSpec]
) -> None:
    dialog = HelpDialog(specs)
    rows = [dialog._list.item(i).text().strip() for i in range(dialog._list.count())]
    # Each unique category contributes a header; each spec contributes a row.
    assert "Source" in rows  # category header
    assert "Filtering" in rows
    assert "Threshold" in rows
    assert any("Gaussian Blur" in r for r in rows)
    dialog.deleteLater()


def test_help_dialog_pins_source_category_to_the_top_of_operations(
    qapp: QApplication, specs: list[OperationSpec]
) -> None:
    dialog = HelpDialog(specs)
    headers = [
        dialog._list.item(i).text().strip()
        for i in range(dialog._list.count())
        if dialog._list.item(i).data(Qt.ItemDataRole.UserRole) is None
    ]
    # "App features" is the top-most header; "Operations" begins the op
    # section, and within ops "Source" must be pinned ahead of the rest.
    assert headers[0] == "App features"
    assert "Operations" in headers
    operations_idx = headers.index("Operations")
    # Source should be the first subheader after the Operations divider.
    assert headers[operations_idx + 1] == "Source"
    dialog.deleteLater()


def test_help_dialog_lists_app_feature_topics(
    qapp: QApplication, specs: list[OperationSpec]
) -> None:
    dialog = HelpDialog(specs)
    titles = [dialog._list.item(i).text() for i in range(dialog._list.count())]
    # A few representative topic titles should appear.
    assert any("Getting started" in t for t in titles)
    assert any("Histogram" in t for t in titles)
    assert any("Region of interest" in t for t in titles)
    assert any("Bulk Export" in t for t in titles)
    assert any("Dataset tab" in t for t in titles)
    dialog.deleteLater()


def test_help_dialog_initial_selection_is_a_real_op(
    qapp: QApplication, specs: list[OperationSpec]
) -> None:
    dialog = HelpDialog(specs)
    current = dialog._list.currentItem()
    assert current is not None
    # The initial pick must be a spec entry, never a category header.
    assert current.data(Qt.ItemDataRole.UserRole) is not None
    dialog.deleteLater()


def test_help_dialog_is_non_modal(qapp: QApplication, specs: list[OperationSpec]) -> None:
    dialog = HelpDialog(specs)
    assert dialog.isModal() is False
    dialog.deleteLater()


def test_selecting_a_spec_loads_its_html_into_the_browser(
    qapp: QApplication,
) -> None:
    real_specs = [_spec("filtering.gaussian_blur", "Gaussian Blur", "Filtering")]
    dialog = HelpDialog(real_specs)
    # Locate the Gaussian Blur leaf and select it explicitly — the default
    # initial selection now lands on the first feature topic.
    for i in range(dialog._list.count()):
        payload = dialog._list.item(i).data(Qt.ItemDataRole.UserRole)
        if payload == "op:filtering.gaussian_blur":
            dialog._list.setCurrentRow(i)
            break
    text = dialog._content.toPlainText()
    assert "Gaussian Blur" in text
    dialog.deleteLater()


def test_selecting_a_feature_topic_loads_its_html(qapp: QApplication, specs: list[OperationSpec]) -> None:
    dialog = HelpDialog(specs)
    for i in range(dialog._list.count()):
        payload = dialog._list.item(i).data(Qt.ItemDataRole.UserRole)
        if payload == "feature:histogram":
            dialog._list.setCurrentRow(i)
            break
    text = dialog._content.toPlainText()
    assert "Histogram" in text
    assert "0-255" in text or "intensity" in text.lower()
    dialog.deleteLater()
