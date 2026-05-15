from __future__ import annotations

from PySide6.QtWidgets import QApplication

from cvstudio.ai import streaming
from cvstudio.core.operation import OperationSpec, Parameter
from cvstudio.core.pipeline import Pipeline
from cvstudio.ui.parameter_panel import ParameterPanel


def _spec_with(*, manual_trigger: bool) -> OperationSpec:
    return OperationSpec(
        id="test.echo",
        name="Echo",
        category="Test",
        description="",
        parameters=(
            Parameter(name="prompt", kind="string", default="hello", step=3),
            Parameter(name="threshold", kind="int", default=5, min=0, max=10),
        ),
        func=lambda image, prompt, threshold: image,  # noqa: ARG005
        manual_trigger=manual_trigger,
    )


def test_run_button_hidden_for_regular_op(qapp: QApplication) -> None:
    panel = ParameterPanel()
    try:
        pipe = Pipeline()
        node = pipe.add(_spec_with(manual_trigger=False))
        panel.set_node(node)
        assert panel._run_button.isVisible() is False
    finally:
        panel.deleteLater()


def test_run_button_visible_for_manual_trigger_op(qapp: QApplication) -> None:
    panel = ParameterPanel()
    panel.show()  # Qt only reports isVisible() correctly once the widget is realized.
    try:
        pipe = Pipeline()
        node = pipe.add(_spec_with(manual_trigger=True))
        panel.set_node(node)
        assert panel._run_button.isVisible() is True
    finally:
        panel.deleteLater()


def test_run_button_emits_run_requested_with_node_id(qapp: QApplication) -> None:
    panel = ParameterPanel()
    try:
        pipe = Pipeline()
        node = pipe.add(_spec_with(manual_trigger=True))
        panel.set_node(node)
        received: list[str] = []
        panel.run_requested.connect(received.append)
        panel._run_button.click()
        assert received == [node.id]
    finally:
        panel.deleteLater()


def test_run_button_commits_pending_string_edit_before_emitting(
    qapp: QApplication,
) -> None:
    """The multi-line StringControl normally commits on focus-out. Clicking
    Run must force-commit so the latest text is in node.params before the
    pipeline re-runs."""
    panel = ParameterPanel()
    try:
        pipe = Pipeline()
        node = pipe.add(_spec_with(manual_trigger=True))
        panel.set_node(node)
        # Simulate the user typing without firing the focus-out commit.
        prompt_control = panel._controls["prompt"]
        prompt_control._edit.setPlainText("a new prompt")  # type: ignore[union-attr]
        # Sanity: control.value() reflects the edit, but node.params has not yet.
        assert prompt_control.value() == "a new prompt"
        assert node.params["prompt"] == "hello"

        panel._run_button.click()
        assert node.params["prompt"] == "a new prompt"
    finally:
        panel.deleteLater()


def test_setting_no_node_hides_run_button(qapp: QApplication) -> None:
    panel = ParameterPanel()
    panel.show()
    try:
        pipe = Pipeline()
        node = pipe.add(_spec_with(manual_trigger=True))
        panel.set_node(node)
        panel.set_node(None)
        assert panel._run_button.isVisible() is False
    finally:
        panel.deleteLater()


def test_response_panel_visible_when_display_text_present(qapp: QApplication) -> None:
    streaming.reset()
    panel = ParameterPanel()
    panel.show()
    try:
        pipe = Pipeline()
        node = pipe.add(_spec_with(manual_trigger=True))
        streaming.set_node_display(node.id, "a tabby cat on a wooden floor")
        panel.set_node(node)
        assert panel._response_view.isVisible() is True
        assert panel._response_view.toPlainText() == "a tabby cat on a wooden floor"
    finally:
        panel.deleteLater()
        streaming.reset()


def test_response_panel_hidden_for_non_manual_trigger_op(qapp: QApplication) -> None:
    streaming.reset()
    panel = ParameterPanel()
    panel.show()
    try:
        pipe = Pipeline()
        node = pipe.add(_spec_with(manual_trigger=False))
        # Even if something tries to publish a display, the panel must stay
        # hidden for regular ops — keeps the right pane uncluttered.
        streaming.set_node_display(node.id, "should-not-show")
        panel.set_node(node)
        assert panel._response_view.isVisible() is False
    finally:
        panel.deleteLater()
        streaming.reset()


def test_response_panel_refreshes_on_bus_progress(qapp: QApplication) -> None:
    """Background workers signal `streaming.bus().progress` after every
    state change; the panel must re-read its node's display text."""
    streaming.reset()
    panel = ParameterPanel()
    panel.show()
    try:
        pipe = Pipeline()
        node = pipe.add(_spec_with(manual_trigger=True))
        panel.set_node(node)
        assert panel._response_view.isVisible() is False  # empty initially

        # Simulate a worker landing a partial.
        streaming.set_node_display(node.id, "Thinking…")
        qapp.processEvents()  # deliver the queued signal
        assert panel._response_view.isVisible() is True
        assert panel._response_view.toPlainText() == "Thinking…"

        streaming.set_node_display(node.id, "final answer")
        qapp.processEvents()
        assert panel._response_view.toPlainText() == "final answer"
    finally:
        panel.deleteLater()
        streaming.reset()
