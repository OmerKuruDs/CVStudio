"""ParameterPanel — right-hand pane that edits the selected node's parameters.

A panel is bound to one PipelineNode at a time. When any control fires
`value_changed`, the panel writes the new value back into the node's params
dict and emits `params_changed` so the MainWindow can re-run the pipeline.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFormLayout, QLabel, QVBoxLayout, QWidget

from cvsandbox.core.pipeline import PipelineNode
from cvsandbox.ui.parameter_widgets import ParameterControl, create_control


class ParameterPanel(QWidget):
    params_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._node: PipelineNode | None = None
        self._controls: dict[str, ParameterControl] = {}

        self._title = QLabel("No operation selected", self)
        self._title.setStyleSheet("font-weight: bold;")

        self._form_host = QWidget(self)
        self._form = QFormLayout(self._form_host)
        self._form.setContentsMargins(0, 0, 0, 0)

        outer = QVBoxLayout(self)
        outer.addWidget(self._title)
        outer.addWidget(self._form_host)
        outer.addStretch(1)

    def set_node(self, node: PipelineNode | None) -> None:
        self._clear_form()
        self._node = node
        if node is None:
            self._title.setText("No operation selected")
            return
        self._title.setText(node.spec.name)
        for param in node.spec.parameters:
            control = create_control(param, self._form_host)
            control.set_value(node.params[param.name])
            control.value_changed.connect(lambda name=param.name: self._on_changed(name))
            self._form.addRow(param.display_label, control)
            self._controls[param.name] = control

    def _on_changed(self, name: str) -> None:
        if self._node is None:
            return
        self._node.params[name] = self._controls[name].value()
        self.params_changed.emit()

    def _clear_form(self) -> None:
        while self._form.rowCount() > 0:
            self._form.removeRow(0)
        self._controls.clear()
