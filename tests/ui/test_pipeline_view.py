from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication

from cvsandbox.core.operation import OperationSpec, Parameter
from cvsandbox.core.pipeline import Pipeline
from cvsandbox.ui.pipeline_view import PipelineView, format_duration


def _noop_spec(spec_id: str, name: str) -> OperationSpec:
    return OperationSpec(
        id=spec_id,
        name=name,
        category="Test",
        description="",
        parameters=(Parameter(name="x", kind="int", default=0, min=0, max=10),),
        func=lambda image, x: image,
    )


@pytest.fixture
def view(qapp: QApplication) -> PipelineView:
    pipeline = Pipeline()
    pipeline.add(_noop_spec("test.a", "Alpha"))
    pipeline.add(_noop_spec("test.b", "Beta"))
    pipeline.add(_noop_spec("test.c", "Gamma"))
    widget = PipelineView(pipeline)
    widget.resize(300, 200)
    return widget


def test_format_duration_uses_microseconds_for_sub_millisecond() -> None:
    assert format_duration(0.000_250) == "250 µs"
    assert format_duration(0.000_001) == "1 µs"


def test_format_duration_uses_milliseconds_in_normal_range() -> None:
    assert format_duration(0.012_3) == "12.3 ms"
    assert format_duration(0.999) == "999.0 ms"


def test_format_duration_uses_seconds_when_slow() -> None:
    assert format_duration(1.234) == "1.23 s"
    assert format_duration(42.0) == "42.00 s"


def test_default_rows_show_no_timing_suffix(view: PipelineView) -> None:
    assert view._list.item(0).text() == "Alpha"
    assert view._list.item(1).text() == "Beta"
    assert view._list.item(2).text() == "Gamma"


def test_set_timings_appends_formatted_suffix(view: PipelineView) -> None:
    view.set_timings([0.001_5, 0.000_400, None])
    assert "1.5 ms" in view._list.item(0).text()
    assert "400 µs" in view._list.item(1).text()
    assert view._list.item(2).text() == "Gamma"  # None → no suffix


def test_set_timings_preserves_check_state(view: PipelineView) -> None:
    from PySide6.QtCore import Qt

    view._pipeline.nodes[1].enabled = False
    view.refresh()
    assert view._list.item(1).checkState() == Qt.CheckState.Unchecked

    view.set_timings([0.001, 0.002, 0.003])
    # Beta is disabled in the pipeline but set_timings should not flip the box.
    assert view._list.item(1).checkState() == Qt.CheckState.Unchecked
    assert view._list.item(0).checkState() == Qt.CheckState.Checked


def test_clear_timings_removes_suffix(view: PipelineView) -> None:
    view.set_timings([0.001, 0.002, 0.003])
    assert "ms" in view._list.item(0).text()
    view.clear_timings()
    assert view._list.item(0).text() == "Alpha"


def test_refresh_after_pipeline_shrinks_drops_excess_timings(view: PipelineView) -> None:
    view.set_timings([0.001, 0.002, 0.003])
    view._pipeline.remove(2)
    view.refresh()
    assert view._list.count() == 2
    # No crash and only 2 timings retained.
    assert len(view._timings) == 2


def test_timing_survives_a_full_refresh(view: PipelineView) -> None:
    view.set_timings([0.005, None, 0.010])
    view.refresh()  # e.g. user reordered → list rebuilt
    assert "5.0 ms" in view._list.item(0).text()
    assert "10.0 ms" in view._list.item(2).text()
    assert view._list.item(1).text() == "Beta"


