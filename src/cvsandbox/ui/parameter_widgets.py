"""ParameterControl widgets — one editor per Parameter kind.

Each control is a small QWidget that knows how to render and edit a single
parameter value. They all share the same shape:

    .value()           -> the current Python value
    .set_value(v)      -> programmatic update (does NOT emit value_changed)
    value_changed      -> Signal emitted on user-driven changes

`create_control(param)` returns the right concrete control for the parameter
kind. UI code never has to switch on the kind itself.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QSpinBox,
    QWidget,
)

from cvsandbox.core.operation import Parameter


class ParameterControl(QWidget):
    """Base class. Subclasses implement value()/set_value() and emit value_changed.

    All subclasses share the same constructor signature `(param, parent)` so the
    factory can instantiate any of them uniformly.
    """

    value_changed = Signal()

    def __init__(self, param: Parameter, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        del param  # base class does no setup; subclasses build their widgets

    def value(self) -> Any:
        raise NotImplementedError

    def set_value(self, value: Any) -> None:
        raise NotImplementedError


class IntControl(ParameterControl):
    def __init__(self, param: Parameter, parent: QWidget | None = None) -> None:
        super().__init__(param, parent)
        assert param.min is not None and param.max is not None
        self._spin = QSpinBox(self)
        self._spin.setRange(int(param.min), int(param.max))
        if param.step is not None:
            self._spin.setSingleStep(int(param.step))
        self._spin.setValue(int(param.default))
        self._spin.valueChanged.connect(lambda _: self.value_changed.emit())
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._spin)

    def value(self) -> int:
        return int(self._spin.value())

    def set_value(self, value: Any) -> None:
        self._spin.blockSignals(True)
        self._spin.setValue(int(value))
        self._spin.blockSignals(False)


class FloatControl(ParameterControl):
    def __init__(self, param: Parameter, parent: QWidget | None = None) -> None:
        super().__init__(param, parent)
        assert param.min is not None and param.max is not None
        self._spin = QDoubleSpinBox(self)
        self._spin.setRange(float(param.min), float(param.max))
        step = float(param.step) if param.step is not None else 0.1
        self._spin.setSingleStep(step)
        self._spin.setDecimals(_decimals_for_step(step))
        self._spin.setValue(float(param.default))
        self._spin.valueChanged.connect(lambda _: self.value_changed.emit())
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._spin)

    def value(self) -> float:
        return float(self._spin.value())

    def set_value(self, value: Any) -> None:
        self._spin.blockSignals(True)
        self._spin.setValue(float(value))
        self._spin.blockSignals(False)


class BoolControl(ParameterControl):
    def __init__(self, param: Parameter, parent: QWidget | None = None) -> None:
        super().__init__(param, parent)
        self._check = QCheckBox(self)
        self._check.setChecked(bool(param.default))
        self._check.stateChanged.connect(lambda _: self.value_changed.emit())
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._check)
        layout.addStretch(1)

    def value(self) -> bool:
        return bool(self._check.isChecked())

    def set_value(self, value: Any) -> None:
        self._check.blockSignals(True)
        self._check.setChecked(bool(value))
        self._check.blockSignals(False)


class ChoiceControl(ParameterControl):
    def __init__(self, param: Parameter, parent: QWidget | None = None) -> None:
        super().__init__(param, parent)
        assert param.choices is not None
        self._combo = QComboBox(self)
        self._combo.addItems(list(param.choices))
        default_idx = self._combo.findText(str(param.default))
        if default_idx >= 0:
            self._combo.setCurrentIndex(default_idx)
        self._combo.currentIndexChanged.connect(lambda _: self.value_changed.emit())
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._combo)

    def value(self) -> str:
        return str(self._combo.currentText())

    def set_value(self, value: Any) -> None:
        idx = self._combo.findText(str(value))
        if idx < 0:
            return
        self._combo.blockSignals(True)
        self._combo.setCurrentIndex(idx)
        self._combo.blockSignals(False)


class KernelSizeControl(ParameterControl):
    """Like IntControl but snaps to odd numbers — cv2 kernel sizes must be odd."""

    def __init__(self, param: Parameter, parent: QWidget | None = None) -> None:
        super().__init__(param, parent)
        assert param.min is not None and param.max is not None
        self._spin = QSpinBox(self)
        self._spin.setRange(int(param.min), int(param.max))
        self._spin.setSingleStep(2)
        self._spin.setValue(_snap_odd(int(param.default)))
        self._spin.valueChanged.connect(self._on_spin_changed)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._spin)

    def _on_spin_changed(self, raw: int) -> None:
        snapped = _snap_odd(int(raw))
        if snapped != raw:
            self._spin.blockSignals(True)
            self._spin.setValue(snapped)
            self._spin.blockSignals(False)
        self.value_changed.emit()

    def value(self) -> int:
        return _snap_odd(int(self._spin.value()))

    def set_value(self, value: Any) -> None:
        self._spin.blockSignals(True)
        self._spin.setValue(_snap_odd(int(value)))
        self._spin.blockSignals(False)


_CONTROLS: dict[str, type[ParameterControl]] = {
    "int": IntControl,
    "float": FloatControl,
    "bool": BoolControl,
    "choice": ChoiceControl,
    "kernel_size": KernelSizeControl,
}


def create_control(param: Parameter, parent: QWidget | None = None) -> ParameterControl:
    """Return a ParameterControl matching the parameter's kind."""
    try:
        control_cls = _CONTROLS[param.kind]
    except KeyError as e:
        raise ValueError(f"No control registered for parameter kind: {param.kind!r}") from e
    return control_cls(param, parent)


def _snap_odd(value: int) -> int:
    return value | 1


def _decimals_for_step(step: float) -> int:
    """Pick a decimal count for a QDoubleSpinBox from its step.

    0.1 -> 1, 0.01 -> 2, etc. Clamped to [0, 6]. We intentionally don't read
    from the parameter's actual values because step is the user-visible
    granularity that matters in the UI.
    """
    if step >= 1:
        return 0
    decimals = 0
    probe = step
    while probe < 1 and decimals < 6:
        probe *= 10
        decimals += 1
    return decimals
