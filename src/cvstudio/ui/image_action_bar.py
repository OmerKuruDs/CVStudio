"""ImageActionBar — horizontal strip of high-traffic actions sitting directly
below the image view.

The previous flow forced users to hunt through File → ... for everyday tasks
like "save the image I just processed". This bar surfaces the I/O actions
right next to where the user looks: load input, save output, record video.
Each button wraps an existing QAction via `setDefaultAction`, so the menu
remains the source of truth for shortcuts / labels.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QFrame, QHBoxLayout, QToolButton, QWidget

BUTTON_HEIGHT = 30


def _button(action: QAction, *, primary: bool = False) -> QToolButton:
    button = QToolButton()
    button.setDefaultAction(action)
    button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
    button.setMinimumHeight(BUTTON_HEIGHT)
    if primary:
        # Picked up by the global stylesheet's [role="primary"] selector.
        button.setProperty("role", "primary")
    return button


def _vertical_separator(parent: QWidget) -> QFrame:
    line = QFrame(parent)
    line.setFrameShape(QFrame.Shape.VLine)
    line.setFrameShadow(QFrame.Shadow.Sunken)
    return line


class ImageActionBar(QWidget):
    def __init__(
        self,
        *,
        open_image_action: QAction,
        open_dataset_action: QAction,
        open_camera_action: QAction,
        open_video_action: QAction,
        save_image_action: QAction,
        record_action: QAction,
        stop_recording_action: QAction,
        stop_capture_action: QAction,
        pause_capture_action: QAction,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("imageActionBar")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(8)

        # Input side: where the source comes from.
        layout.addWidget(_button(open_image_action))
        layout.addWidget(_button(open_dataset_action))
        layout.addWidget(_button(open_camera_action))
        layout.addWidget(_button(open_video_action))
        layout.addWidget(_button(pause_capture_action))
        layout.addWidget(_button(stop_capture_action))
        layout.addWidget(_vertical_separator(self))

        # Output side: prominent Save, plus recording controls.
        layout.addWidget(_button(save_image_action, primary=True))
        layout.addWidget(_button(record_action))
        layout.addWidget(_button(stop_recording_action))

        layout.addStretch(1)
