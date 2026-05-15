"""HelpDialog — non-modal browser of in-app documentation.

The left pane has two scrollable sections:

  1. <b>App features</b> — high-level guides for the things you click /
     toggle (Open Image, the pipeline editor, histogram, ROI, saving, …).
  2. <b>Operations</b> — one entry per registered op, grouped by category.

The right pane renders the rich HTML blob for whichever entry is selected.
The dialog is non-modal so it can sit next to the main editor while the
user keeps tuning.
"""

from __future__ import annotations

from collections.abc import Iterable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from cvstudio.core.op_docs import FEATURE_TOPICS, get_feature_doc, get_op_doc
from cvstudio.core.operation import OperationSpec

_FEATURE_PREFIX = "feature:"
_OP_PREFIX = "op:"


def _sort_specs(specs: Iterable[OperationSpec]) -> list[OperationSpec]:
    def key(spec: OperationSpec) -> tuple[int, str, str]:
        category_priority = 0 if spec.category == "Source" else 1
        return (category_priority, spec.category, spec.name)

    return sorted(specs, key=key)


class HelpDialog(QDialog):
    def __init__(self, specs: Iterable[OperationSpec], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Operation Guide — CVStudio")
        self.setModal(False)
        self.resize(980, 660)

        intro = QLabel(
            "Pick a topic on the left. Application features explain how the "
            "main window works; the Operations section documents every op "
            "you can drop into a pipeline.",
            self,
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #475569;")

        self._list = QListWidget(self)
        self._list.setMinimumWidth(260)
        self._list.currentItemChanged.connect(self._on_select)

        self._content = QTextBrowser(self)
        self._content.setOpenExternalLinks(True)
        self._content.document().setDefaultStyleSheet(
            "h2 { color: #0e7490; margin-bottom: 4px; }"
            "h3 { color: #0b1437; margin-top: 14px; margin-bottom: 4px; }"
            "p { color: #0b1437; line-height: 1.45; }"
            "li { color: #0b1437; margin-bottom: 4px; }"
            "code { background-color: #f1f5f9; padding: 1px 4px; "
            "border-radius: 3px; color: #0e7490; }"
        )

        self._populate_features()
        self._populate_operations(specs)
        self._select_first_real_entry()

        split = QHBoxLayout()
        split.addWidget(self._list, 1)
        split.addWidget(self._content, 3)

        layout = QVBoxLayout(self)
        layout.addWidget(intro)
        layout.addLayout(split)

    # ------------------------------------------------------------------ build

    def _populate_features(self) -> None:
        self._add_header("App features")
        for topic in FEATURE_TOPICS:
            self._add_leaf(topic.title, f"{_FEATURE_PREFIX}{topic.key}")

    def _populate_operations(self, specs: Iterable[OperationSpec]) -> None:
        self._add_header("Operations")
        current_category = ""
        for spec in _sort_specs(specs):
            if spec.category != current_category:
                self._add_subheader(spec.category)
                current_category = spec.category
            self._add_leaf(f"    {spec.name}", f"{_OP_PREFIX}{spec.id}")

    def _add_header(self, text: str) -> None:
        item = QListWidgetItem(text)
        item.setFlags(Qt.ItemFlag.ItemIsEnabled)
        font = item.font()
        font.setBold(True)
        font.setPointSize(font.pointSize() + 1)
        item.setFont(font)
        item.setData(Qt.ItemDataRole.UserRole, None)
        item.setForeground(Qt.GlobalColor.darkCyan)
        self._list.addItem(item)

    def _add_subheader(self, text: str) -> None:
        item = QListWidgetItem(text)
        item.setFlags(Qt.ItemFlag.ItemIsEnabled)
        font = item.font()
        font.setBold(True)
        item.setFont(font)
        item.setData(Qt.ItemDataRole.UserRole, None)
        self._list.addItem(item)

    def _add_leaf(self, text: str, payload: str) -> None:
        item = QListWidgetItem(text)
        item.setData(Qt.ItemDataRole.UserRole, payload)
        self._list.addItem(item)

    def _select_first_real_entry(self) -> None:
        for i in range(self._list.count()):
            if self._list.item(i).data(Qt.ItemDataRole.UserRole) is not None:
                self._list.setCurrentRow(i)
                break

    # ------------------------------------------------------------------ events

    def _on_select(
        self, current: QListWidgetItem | None, _previous: QListWidgetItem | None
    ) -> None:
        if current is None:
            return
        payload = current.data(Qt.ItemDataRole.UserRole)
        if not isinstance(payload, str):
            return
        if payload.startswith(_FEATURE_PREFIX):
            self._content.setHtml(get_feature_doc(payload[len(_FEATURE_PREFIX):]))
        elif payload.startswith(_OP_PREFIX):
            self._content.setHtml(get_op_doc(payload[len(_OP_PREFIX):]))
