"""ImageToolsPanel — vertical sidebar of toggle / action buttons that mirror
the View menu.

Each button is wired to an existing QAction via `setDefaultAction`, so the
menu, keyboard shortcuts, and the sidebar all share the same checked state
and click behaviour. No new logic lives in this widget — it is purely a more
discoverable surface for the actions defined on MainWindow.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QGroupBox,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

PANEL_WIDTH = 180
BUTTON_HEIGHT = 32


def _make_button(
    action: QAction, parent: QWidget, *, label: str | None = None
) -> QToolButton:
    """Wrap a QAction in a sidebar button. `label` lets the caller override
    the action's verbose menu text with a tighter sidebar caption so it fits
    inside the narrow column — the action still owns checked state, shortcut,
    and click behaviour."""
    button = QToolButton(parent)
    button.setDefaultAction(action)
    button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
    button.setMinimumHeight(BUTTON_HEIGHT)
    button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    if label is not None:
        button.setText(label)
        # Keep the action's verbose description available on hover.
        button.setToolTip(action.text().replace("&", ""))
    return button


class ImageToolsPanel(QWidget):
    def __init__(
        self,
        *,
        split_action: QAction,
        downscale_action: QAction,
        select_roi_action: QAction,
        clear_roi_action: QAction,
        randomize_paste_action: QAction,
        clear_paste_action: QAction,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setFixedWidth(PANEL_WIDTH)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        # Sidebar captions stay terse so they fit the narrow column; the menu
        # actions keep their full descriptive titles for the File menu.
        view_group = QGroupBox("View", self)
        view_layout = QVBoxLayout(view_group)
        view_layout.setSpacing(6)
        view_layout.addWidget(_make_button(split_action, view_group, label="Split view"))
        view_layout.addWidget(
            _make_button(downscale_action, view_group, label="Downscale previews")
        )
        layout.addWidget(view_group)

        roi_group = QGroupBox("ROI", self)
        roi_layout = QVBoxLayout(roi_group)
        roi_layout.setSpacing(6)
        roi_layout.addWidget(_make_button(select_roi_action, roi_group, label="Select ROI"))
        roi_layout.addWidget(_make_button(clear_roi_action, roi_group, label="Clear ROI"))
        layout.addWidget(roi_group)

        paste_group = QGroupBox("Paste destination", self)
        paste_layout = QVBoxLayout(paste_group)
        paste_layout.setSpacing(6)
        paste_layout.addWidget(
            _make_button(randomize_paste_action, paste_group, label="Randomize position")
        )
        paste_layout.addWidget(
            _make_button(clear_paste_action, paste_group, label="Clear position")
        )
        layout.addWidget(paste_group)

        layout.addStretch(1)
