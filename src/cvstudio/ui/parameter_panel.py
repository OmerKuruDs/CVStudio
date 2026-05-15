"""ParameterPanel — right-hand pane that edits the selected node's parameters.

A panel is bound to one PipelineNode at a time. When any control fires
`value_changed`, the panel writes the new value back into the node's params
dict and emits `params_changed` so the MainWindow can re-run the pipeline.

Manual-trigger ops (`spec.manual_trigger=True`, e.g. the VLM Q&A node)
also get a Run button at the bottom of the form. Clicking it commits
any in-progress edits, then emits `run_requested(node_id)` — MainWindow
authorizes that node so the next pipeline run is allowed to spawn the
expensive backend call (VLM inference, classifier, ...).
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFormLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from cvstudio.ai.streaming import bus as streaming_bus
from cvstudio.ai.streaming import get_node_display
from cvstudio.core.pipeline import PipelineNode
from cvstudio.ui.parameter_widgets import (
    ParameterControl,
    StringControl,
    create_control,
)


class ParameterPanel(QWidget):
    params_changed = Signal()
    run_requested = Signal(str)
    """Emitted with the node id when the user clicks Run on a manual-
    trigger op. MainWindow uses it to authorize the next spawn."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._node: PipelineNode | None = None
        self._controls: dict[str, ParameterControl] = {}

        self._title = QLabel("No operation selected", self)
        self._title.setStyleSheet("font-weight: bold;")

        self._form_host = QWidget(self)
        self._form = QFormLayout(self._form_host)
        self._form.setContentsMargins(0, 0, 0, 0)

        self._run_button = QPushButton("▶ Run", self)
        self._run_button.setProperty("role", "primary")
        self._run_button.setMinimumHeight(32)
        self._run_button.clicked.connect(self._on_run_clicked)
        self._run_button.setVisible(False)

        # AI response area — populated by manual_trigger ops via
        # streaming.set_node_display. Kept here (rather than overlaying
        # text on the image) so the image stays clean for downstream
        # OpenCV ops, and long replies aren't clipped at the bottom.
        self._response_header = QLabel("AI Response", self)
        self._response_header.setStyleSheet(
            "color: #64748b; font-size: 9pt; padding-top: 8px;"
        )
        self._response_header.setVisible(False)
        self._response_view = QTextEdit(self)
        self._response_view.setReadOnly(True)
        self._response_view.setMinimumHeight(80)
        self._response_view.setStyleSheet(
            "QTextEdit { background-color: rgba(0,0,0,0.04);"
            " border: 1px solid rgba(0,0,0,0.08); border-radius: 4px;"
            " padding: 6px; }"
        )
        self._response_view.setVisible(False)

        outer = QVBoxLayout(self)
        outer.addWidget(self._title)
        outer.addWidget(self._form_host)
        outer.addWidget(self._run_button)
        outer.addWidget(self._response_header)
        outer.addWidget(self._response_view, 1)

        # Background AI workers fire this signal as state changes; we
        # refresh whatever response text the selected node currently has.
        streaming_bus().progress.connect(self._refresh_response)

    def set_node(self, node: PipelineNode | None) -> None:
        self._clear_form()
        self._node = node
        if node is None:
            self._title.setText("No operation selected")
            self._run_button.setVisible(False)
            self._set_response_visible(False)
            return
        self._title.setText(node.spec.name)
        for param in node.spec.parameters:
            control = create_control(param, self._form_host)
            control.set_value(node.params[param.name])
            control.value_changed.connect(lambda name=param.name: self._on_changed(name))
            self._form.addRow(param.display_label, control)
            self._controls[param.name] = control
        self._run_button.setVisible(bool(node.spec.manual_trigger))
        self._refresh_response()

    def _on_changed(self, name: str) -> None:
        if self._node is None:
            return
        self._node.params[name] = self._controls[name].value()
        self.params_changed.emit()

    def _on_run_clicked(self) -> None:
        if self._node is None:
            return
        # Force-commit any in-progress edits — most controls already wrote
        # to node.params on their value_changed, but multi-line text
        # commits only on focus-out / Ctrl+Enter and the user clicking Run
        # has not necessarily left the field.
        for name, control in self._controls.items():
            if isinstance(control, StringControl):
                control.commit_pending_edit()
            self._node.params[name] = control.value()
        self.run_requested.emit(self._node.id)
        self.params_changed.emit()

    def _clear_form(self) -> None:
        while self._form.rowCount() > 0:
            self._form.removeRow(0)
        self._controls.clear()

    def _refresh_response(self) -> None:
        """Pull the current display text for the selected node and toggle
        the response area's visibility accordingly. No-op for nodes that
        do not declare `manual_trigger`."""
        if self._node is None or not self._node.spec.manual_trigger:
            self._set_response_visible(False)
            return
        text = get_node_display(self._node.id) or ""
        if text:
            # `setPlainText` instead of `setText` so embedded HTML in
            # responses stays literal — AI outputs are untrusted.
            if text != self._response_view.toPlainText():
                self._response_view.setPlainText(text)
            self._set_response_visible(True)
        else:
            self._set_response_visible(False)

    def _set_response_visible(self, visible: bool) -> None:
        self._response_header.setVisible(visible)
        self._response_view.setVisible(visible)
