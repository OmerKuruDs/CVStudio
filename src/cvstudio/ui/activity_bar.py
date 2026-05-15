"""ActivityBar — vertical mode selector for the main window.

A thin column of toggle buttons sitting on the left edge of the Editor tab.
Each button represents a top-level mode (Op / 2D / 3D); switching modes
swaps which page of the central QStackedWidget is shown.

The bar can be collapsed via the chevron button at the bottom. When
collapsed, each button shrinks to a compact single-character label
("O" / "2" / "3") so it still survives mouse-click but takes minimum
horizontal space.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


class ActivityBar(QWidget):
    MODE_OP = "Op"
    MODE_2D = "2D"
    MODE_3D = "3D"
    MODE_AI = "AI"

    _LABELS_FULL: dict[str, str] = {
        MODE_OP: "Op",
        MODE_2D: "2D",
        MODE_3D: "3D",
        MODE_AI: "AI",
    }
    _LABELS_COMPACT: dict[str, str] = {
        MODE_OP: "O",
        MODE_2D: "2",
        MODE_3D: "3",
        MODE_AI: "A",
    }

    # Pixel widths chosen so collapsed mode is roughly half the expanded width
    # — enough to stay clickable, narrow enough to feel like an "icon rail".
    _WIDTH_EXPANDED = 56
    _WIDTH_COLLAPSED = 28

    mode_changed = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("ActivityBar")
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self._collapsed = False

        self._buttons: dict[str, QToolButton] = {}
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 4, 2, 4)
        layout.setSpacing(2)

        for mode in (self.MODE_OP, self.MODE_2D, self.MODE_3D, self.MODE_AI):
            btn = QToolButton(self)
            btn.setText(self._LABELS_FULL[mode])
            btn.setCheckable(True)
            btn.setToolButtonStyle(self._ToolButtonStyleText())
            btn.setProperty("mode", mode)
            btn.setMinimumHeight(44)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            btn.clicked.connect(lambda _checked=False, m=mode: self._activate(m))
            layout.addWidget(btn)
            self._buttons[mode] = btn
            self._group.addButton(btn)

        layout.addStretch(1)

        # Bottom collapse toggle. Distinct from the mode buttons so it
        # doesn't enter the exclusive group.
        self._collapse_btn = QToolButton(self)
        self._collapse_btn.setText("<<")
        self._collapse_btn.setToolTip("Collapse / expand the activity bar")
        self._collapse_btn.setMinimumHeight(28)
        self._collapse_btn.clicked.connect(self.toggle_collapsed)
        layout.addWidget(self._collapse_btn)

        # Default to Op mode selected; emit mode_changed once during
        # construction so MainWindow can sync its stacked widget.
        self._buttons[self.MODE_OP].setChecked(True)
        self.setFixedWidth(self._WIDTH_EXPANDED)

    # ------------------------------------------------------------------ public

    @property
    def current_mode(self) -> str:
        for mode, btn in self._buttons.items():
            if btn.isChecked():
                return mode
        return self.MODE_OP  # fallback — should not occur

    def set_current_mode(self, mode: str) -> None:
        if mode not in self._buttons:
            return
        self._buttons[mode].setChecked(True)
        self.mode_changed.emit(mode)

    def toggle_collapsed(self) -> None:
        self._collapsed = not self._collapsed
        labels = self._LABELS_COMPACT if self._collapsed else self._LABELS_FULL
        for mode, btn in self._buttons.items():
            btn.setText(labels[mode])
        self._collapse_btn.setText(">>" if self._collapsed else "<<")
        self.setFixedWidth(self._WIDTH_COLLAPSED if self._collapsed else self._WIDTH_EXPANDED)

    @property
    def is_collapsed(self) -> bool:
        return self._collapsed

    # ----------------------------------------------------------------- internal

    def _activate(self, mode: str) -> None:
        # Mode buttons are exclusive; we still emit on every click so a
        # user re-clicking the active mode triggers a refresh (e.g. to
        # reset the stack to its default state).
        self.mode_changed.emit(mode)

    def _ToolButtonStyleText(self):  # noqa: N802 (PySide style)
        # Sticking with text labels keeps the bar usable on any system
        # without bundling icon assets. ToolButtonTextOnly is enum value 1
        # on every Qt 6 release we care about.
        from PySide6.QtCore import Qt as _Qt

        return _Qt.ToolButtonStyle.ToolButtonTextOnly
