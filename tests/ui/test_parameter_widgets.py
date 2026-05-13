from __future__ import annotations

from PySide6.QtWidgets import QApplication

from cvsandbox.core.operation import Parameter
from cvsandbox.ui.parameter_widgets import (
    BoolControl,
    ChoiceControl,
    FloatControl,
    IntControl,
    KernelSizeControl,
    create_control,
)


def test_int_control_reports_default(qapp: QApplication) -> None:
    param = Parameter(name="x", kind="int", default=7, min=0, max=100)
    ctrl = create_control(param)
    assert isinstance(ctrl, IntControl)
    assert ctrl.value() == 7


def test_float_control_reports_default(qapp: QApplication) -> None:
    param = Parameter(name="x", kind="float", default=1.5, min=0.0, max=10.0, step=0.1)
    ctrl = create_control(param)
    assert isinstance(ctrl, FloatControl)
    assert ctrl.value() == 1.5


def test_bool_control_reports_default(qapp: QApplication) -> None:
    param = Parameter(name="x", kind="bool", default=True)
    ctrl = create_control(param)
    assert isinstance(ctrl, BoolControl)
    assert ctrl.value() is True


def test_choice_control_picks_default(qapp: QApplication) -> None:
    param = Parameter(name="x", kind="choice", default="b", choices=("a", "b", "c"))
    ctrl = create_control(param)
    assert isinstance(ctrl, ChoiceControl)
    assert ctrl.value() == "b"


def test_kernel_size_snaps_even_default_to_odd(qapp: QApplication) -> None:
    param = Parameter(name="ksize", kind="kernel_size", default=4, min=1, max=99, step=2)
    ctrl = create_control(param)
    assert isinstance(ctrl, KernelSizeControl)
    assert ctrl.value() == 5


def test_set_value_round_trip(qapp: QApplication) -> None:
    param = Parameter(name="x", kind="int", default=0, min=0, max=100)
    ctrl = create_control(param)
    ctrl.set_value(42)
    assert ctrl.value() == 42


def test_value_changed_fires_on_user_update(qapp: QApplication) -> None:
    param = Parameter(name="x", kind="int", default=0, min=0, max=100)
    ctrl = create_control(param)
    received: list[int] = []
    ctrl.value_changed.connect(lambda: received.append(ctrl.value()))
    ctrl._spin.setValue(10)  # type: ignore[attr-defined]
    assert received == [10]


def test_set_value_does_not_emit(qapp: QApplication) -> None:
    param = Parameter(name="x", kind="int", default=0, min=0, max=100)
    ctrl = create_control(param)
    received: list[int] = []
    ctrl.value_changed.connect(lambda: received.append(ctrl.value()))
    ctrl.set_value(50)
    assert received == []
