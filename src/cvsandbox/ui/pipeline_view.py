"""PipelineView — bottom strip showing the current Pipeline as a list.

Each row = one PipelineNode with an enable checkbox and the operation's name.
The user can select a row (to edit its parameters in the right panel), reorder
via Up/Down buttons, and remove a row.

Signals:
- `selection_changed(int)` — index of the selected node, -1 if none
- `pipeline_changed()`     — structure or enabled state changed
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from cvsandbox.core.pipeline import Pipeline


class PipelineView(QWidget):
    selection_changed = Signal(int)
    pipeline_changed = Signal()

    def __init__(self, pipeline: Pipeline, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pipeline = pipeline

        self._list = QListWidget(self)
        self._list.currentRowChanged.connect(self.selection_changed.emit)
        self._list.itemChanged.connect(self._on_item_changed)

        self._up = QPushButton("↑", self)
        self._down = QPushButton("↓", self)
        self._remove = QPushButton("Remove", self)
        self._up.clicked.connect(self._on_up)
        self._down.clicked.connect(self._on_down)
        self._remove.clicked.connect(self._on_remove)

        buttons = QVBoxLayout()
        buttons.addWidget(self._up)
        buttons.addWidget(self._down)
        buttons.addWidget(self._remove)
        buttons.addStretch(1)

        outer = QHBoxLayout(self)
        outer.addWidget(self._list, 1)
        outer.addLayout(buttons)

        self.refresh()

    def refresh(self) -> None:
        self._list.blockSignals(True)
        self._list.clear()
        for node in self._pipeline.nodes:
            item = QListWidgetItem(node.spec.name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if node.enabled else Qt.CheckState.Unchecked)
            self._list.addItem(item)
        self._list.blockSignals(False)

    def select(self, index: int) -> None:
        self._list.setCurrentRow(index)

    def _on_item_changed(self, item: QListWidgetItem) -> None:
        index = self._list.row(item)
        if 0 <= index < len(self._pipeline.nodes):
            self._pipeline.nodes[index].enabled = item.checkState() == Qt.CheckState.Checked
            self.pipeline_changed.emit()

    def _on_up(self) -> None:
        index = self._list.currentRow()
        if index <= 0:
            return
        self._pipeline.move(index, index - 1)
        self.refresh()
        self.select(index - 1)
        self.pipeline_changed.emit()

    def _on_down(self) -> None:
        index = self._list.currentRow()
        if index < 0 or index >= len(self._pipeline.nodes) - 1:
            return
        self._pipeline.move(index, index + 1)
        self.refresh()
        self.select(index + 1)
        self.pipeline_changed.emit()

    def _on_remove(self) -> None:
        index = self._list.currentRow()
        if index < 0:
            return
        self._pipeline.remove(index)
        self.refresh()
        new_index = min(index, len(self._pipeline.nodes) - 1)
        self.select(new_index)
        self.pipeline_changed.emit()
