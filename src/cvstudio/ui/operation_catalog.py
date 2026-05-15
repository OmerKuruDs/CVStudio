"""OperationCatalog — left pane listing registered operations grouped by category.

Double-click (or pressing Enter on a leaf) emits `operation_chosen` with the
OperationSpec id. The MainWindow appends a corresponding PipelineNode.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QTreeWidget, QTreeWidgetItem, QWidget

from cvsandbox.core.registry import all_operations


class OperationCatalog(QTreeWidget):
    operation_chosen = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setHeaderLabel("Operations")
        self.setSelectionMode(self.SelectionMode.SingleSelection)
        self.itemActivated.connect(self._on_activated)
        self._category_filter: str | None = None
        self.refresh()

    def set_category_filter(self, category: str | None) -> None:
        """Restrict the visible ops to a single category. Pass None to
        clear the filter and show everything again. Used by the AI
        activity-bar mode to focus the catalog on AI ops without
        building a separate widget."""
        if category == self._category_filter:
            return
        self._category_filter = category
        self.refresh()
        if category is not None:
            self.setHeaderLabel(f"Operations — {category}")
        else:
            self.setHeaderLabel("Operations")

    def refresh(self) -> None:
        self.clear()
        categories: dict[str, QTreeWidgetItem] = {}
        for spec in all_operations():
            if (
                self._category_filter is not None
                and spec.category != self._category_filter
            ):
                continue
            parent = categories.get(spec.category)
            if parent is None:
                parent = QTreeWidgetItem(self, [spec.category])
                parent.setFlags(parent.flags() & ~Qt.ItemFlag.ItemIsSelectable)
                categories[spec.category] = parent
            leaf = QTreeWidgetItem(parent, [spec.name])
            leaf.setData(0, Qt.ItemDataRole.UserRole, spec.id)
            leaf.setToolTip(0, spec.description)
        self.expandAll()

    def _on_activated(self, item: QTreeWidgetItem, _column: int) -> None:
        spec_id = item.data(0, Qt.ItemDataRole.UserRole)
        if isinstance(spec_id, str):
            self.operation_chosen.emit(spec_id)
