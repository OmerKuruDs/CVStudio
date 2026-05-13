"""PipelineView — bottom strip showing the current Pipeline as a list.

Each row = one PipelineNode with an enable checkbox, the operation's name, and
(when available) a per-step timing measured by the preview worker. The user
can select a row (to edit its parameters in the right panel), reorder via
Up/Down buttons, and remove a row.

Signals:
- `selection_changed(int)` — index of the selected node, -1 if none
- `pipeline_changed()`     — structure or enabled state changed
"""

from __future__ import annotations

from collections.abc import Sequence

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


def format_duration(seconds: float) -> str:
    """Format a duration in seconds as a short human-readable string."""
    if seconds < 1e-3:
        return f"{seconds * 1e6:.0f} µs"
    if seconds < 1.0:
        return f"{seconds * 1e3:.1f} ms"
    return f"{seconds:.2f} s"


class PipelineView(QWidget):
    selection_changed = Signal(int)
    pipeline_changed = Signal()

    def __init__(self, pipeline: Pipeline, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pipeline = pipeline
        self._timings: list[float | None] = []

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
        # Truncate stored timings so they never outrun the pipeline length.
        del self._timings[len(self._pipeline.nodes):]
        self._list.blockSignals(True)
        self._list.clear()
        for i, node in enumerate(self._pipeline.nodes):
            item = QListWidgetItem(self._row_text(i, node.spec.name))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if node.enabled else Qt.CheckState.Unchecked)
            self._list.addItem(item)
        self._list.blockSignals(False)

    def select(self, index: int) -> None:
        self._list.setCurrentRow(index)

    def set_timings(self, timings: Sequence[float | None]) -> None:
        """Update per-row timing display. Length must match the pipeline; rows
        whose timing is None show no suffix (e.g. disabled or never executed).
        Calling this does not rebuild the list or affect selection."""
        self._timings = list(timings)
        for i in range(min(self._list.count(), len(self._timings))):
            item = self._list.item(i)
            if item is None:
                continue
            previous_state = item.checkState()
            self._list.blockSignals(True)
            item.setText(self._row_text(i, self._pipeline.nodes[i].spec.name))
            item.setCheckState(previous_state)
            self._list.blockSignals(False)

    def clear_timings(self) -> None:
        self.set_timings([None] * len(self._pipeline.nodes))

    def _row_text(self, index: int, name: str) -> str:
        timing = self._timings[index] if index < len(self._timings) else None
        if timing is None:
            return name
        return f"{name}    {format_duration(timing)}"

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
