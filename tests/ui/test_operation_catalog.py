from __future__ import annotations

from PySide6.QtWidgets import QApplication

from cvstudio.operations import load_builtin_operations
from cvstudio.ui.operation_catalog import OperationCatalog


def _category_items(catalog: OperationCatalog) -> list[str]:
    """Return the names of every top-level category item currently in the tree."""
    return [
        catalog.topLevelItem(i).text(0)
        for i in range(catalog.topLevelItemCount())
        if catalog.topLevelItem(i) is not None
    ]


def test_catalog_shows_every_category_by_default(qapp: QApplication) -> None:
    load_builtin_operations()
    catalog = OperationCatalog()
    try:
        categories = _category_items(catalog)
        assert "AI" in categories
        assert "Filtering" in categories or any("Filter" in c for c in categories)
        assert len(categories) > 1
    finally:
        catalog.deleteLater()


def test_catalog_filter_to_ai_only_shows_ai_category(qapp: QApplication) -> None:
    load_builtin_operations()
    catalog = OperationCatalog()
    try:
        catalog.set_category_filter("AI")
        categories = _category_items(catalog)
        assert categories == ["AI"]
        # Filtered header makes the mode obvious.
        assert "AI" in catalog.headerItem().text(0)
    finally:
        catalog.deleteLater()


def test_catalog_filter_cleared_restores_full_list(qapp: QApplication) -> None:
    load_builtin_operations()
    catalog = OperationCatalog()
    try:
        catalog.set_category_filter("AI")
        catalog.set_category_filter(None)
        categories = _category_items(catalog)
        assert "AI" in categories
        assert len(categories) > 1
        assert catalog.headerItem().text(0) == "Operations"
    finally:
        catalog.deleteLater()
