"""Compact viz-mode pages shown when the activity bar selects 2D or 3D.

Each page composes:

* an ``Add Op`` toolbar button (hierarchical popup of every registered
  operation — replaces the left-side OperationCatalog tree),
* a sub-mode toggle so the user can switch between the two flavours of
  the active dimension (Image / Heatmap for 2D, Surface / Point Cloud
  for 3D) without leaving the page,
* the actual pyqtgraph view widget,
* a compact right rail that hosts mode-specific viz controls
  (colormap, Z scale) followed by a slot where the MainWindow re-parents
  the shared ParameterPanel.

Reuses the mode widget classes (:class:`_ImageMode`, :class:`_HeatmapMode`,
:class:`_SurfaceMode`, :class:`_PointCloudMode`) from
``visualization_panel`` — the new pages just wrap them in the compact
layout the activity-bar UX wants.
"""

from __future__ import annotations

from itertools import groupby

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QSizePolicy,
    QSlider,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from cvsandbox.core.registry import all_operations
from cvsandbox.ui.visualization_panel import (
    DEFAULT_HEATMAP,
    HEATMAP_COLORMAPS,
    _HeatmapMode,
    _ImageMode,
    _PointCloudMode,
    _SurfaceMode,
)


class AddOpButton(QToolButton):
    """Compact replacement for the OperationCatalog tree.

    Click → menu pops up with one submenu per category, leaf actions per
    operation. Selecting a leaf emits ``operation_chosen`` with that op's
    spec id, matching the existing ``OperationCatalog`` signal — so
    MainWindow's add-op handler doesn't need to know who fired it.
    """

    operation_chosen = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setText("➕ Add Op")
        self.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.setToolTip("Browse and add an operation to the pipeline")
        self._menu = QMenu(self)
        self.setMenu(self._menu)
        self.refresh()

    def refresh(self) -> None:
        """Rebuild the popup. Call this if the registry has changed."""
        self._menu.clear()
        specs = sorted(all_operations(), key=lambda s: (s.category, s.name))
        for category, group in groupby(specs, key=lambda s: s.category):
            submenu = self._menu.addMenu(category)
            for spec in group:
                action = QAction(spec.name, submenu)
                action.setToolTip(spec.description)
                spec_id = spec.id
                action.triggered.connect(
                    lambda _checked=False, sid=spec_id: self.operation_chosen.emit(sid)
                )
                submenu.addAction(action)


class _VizPageBase(QWidget):
    """Shared scaffolding for Viz2DPage / Viz3DPage.

    Builds the row that holds AddOpButton + a sub-mode toggle on the left,
    a centred pyqtgraph view in the middle, and a right rail consisting
    of a viz-controls container followed by a slot where MainWindow
    parents the shared ParameterPanel.
    """

    SIDEBAR_WIDTH = 240

    operation_chosen = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._last_image: np.ndarray | None = None
        self._build_layout()

    def _build_layout(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(4)

        # ---- top toolbar
        toolbar = QHBoxLayout()
        toolbar.setSpacing(6)
        self._add_op_btn = AddOpButton(self)
        self._add_op_btn.operation_chosen.connect(self.operation_chosen.emit)
        toolbar.addWidget(self._add_op_btn)
        toolbar.addSpacing(12)
        self._submode_label = QLabel("View:", self)
        toolbar.addWidget(self._submode_label)
        # Sub-mode toggle (Image/Heatmap for 2D, Surface/Cloud for 3D);
        # the concrete subclass populates it.
        self._submode_buttons: dict[str, QPushButton] = {}
        self._submode_group = QButtonGroup(self)
        self._submode_group.setExclusive(True)
        self._toolbar_layout = toolbar
        toolbar.addStretch(1)
        outer.addLayout(toolbar)

        # ---- center + right rail
        body = QHBoxLayout()
        body.setSpacing(4)
        self._view_stack = QStackedWidget(self)
        body.addWidget(self._view_stack, 1)

        rail = QWidget(self)
        rail.setFixedWidth(self.SIDEBAR_WIDTH)
        self._rail_layout = QVBoxLayout(rail)
        self._rail_layout.setContentsMargins(4, 4, 4, 4)
        self._rail_layout.setSpacing(8)
        body.addWidget(rail)

        outer.addLayout(body, 1)

        # Where the shared ParameterPanel gets parented when this page
        # becomes active. Subclasses fill ``_rail_layout`` with their
        # viz controls first, then call ``_install_param_slot`` to add
        # the slot widget after.
        self._param_slot = QWidget(self)
        self._param_slot_layout = QVBoxLayout(self._param_slot)
        self._param_slot_layout.setContentsMargins(0, 0, 0, 0)

    def _install_param_slot(self) -> None:
        separator = QLabel("─── Params ───", self)
        separator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        separator.setStyleSheet("color: #888888;")
        self._rail_layout.addWidget(separator)
        self._rail_layout.addWidget(self._param_slot, 1)
        self._rail_layout.addStretch(1)

    def _add_submode(self, name: str, checked: bool = False) -> QPushButton:
        btn = QPushButton(name, self)
        btn.setCheckable(True)
        btn.setChecked(checked)
        btn.setMinimumWidth(60)
        self._submode_buttons[name] = btn
        self._submode_group.addButton(btn)
        # Insert before the trailing stretch so buttons stay left-aligned.
        # The stretch is the last item added in _build_layout.
        insert_at = self._toolbar_layout.count() - 1
        self._toolbar_layout.insertWidget(insert_at, btn)
        return btn

    # ------------------------------------------------------------------ public

    def set_image(self, image: np.ndarray | None) -> None:
        self._last_image = image

    def clear(self) -> None:
        self.set_image(None)

    def install_param_panel(self, panel: QWidget) -> None:
        """Move the shared ParameterPanel into this page's right-rail slot."""
        panel.setParent(self._param_slot)
        # Strip any previous slot occupant (if same panel was here before
        # Qt is a no-op; otherwise this clears stale children safely).
        while self._param_slot_layout.count() > 0:
            item = self._param_slot_layout.takeAt(0)
            if item is not None and item.widget() is not None:
                item.widget().setParent(None)
        self._param_slot_layout.addWidget(panel)
        panel.show()


class Viz2DPage(_VizPageBase):
    """Activity-bar 2D page — Image or Heatmap sub-modes."""

    SUBMODE_IMAGE = "Image"
    SUBMODE_HEATMAP = "Heatmap"

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._image_widget = _ImageMode(self)
        self._heatmap_widget = _HeatmapMode(self)
        self._view_stack.addWidget(self._image_widget)
        self._view_stack.addWidget(self._heatmap_widget)

        btn_image = self._add_submode(self.SUBMODE_IMAGE, checked=True)
        btn_heatmap = self._add_submode(self.SUBMODE_HEATMAP)
        btn_image.clicked.connect(lambda: self._set_submode(self.SUBMODE_IMAGE))
        btn_heatmap.clicked.connect(lambda: self._set_submode(self.SUBMODE_HEATMAP))

        # Right rail: colormap selector (shown only in Heatmap).
        self._colormap_label = QLabel("Colormap:", self)
        self._colormap_selector = QComboBox(self)
        for friendly in HEATMAP_COLORMAPS:
            self._colormap_selector.addItem(friendly)
        self._colormap_selector.setCurrentText(DEFAULT_HEATMAP)
        self._colormap_selector.currentTextChanged.connect(self._on_colormap_changed)
        self._rail_layout.addWidget(self._colormap_label)
        self._rail_layout.addWidget(self._colormap_selector)

        self._current_submode = self.SUBMODE_IMAGE
        self._install_param_slot()
        self._update_controls_visibility()

    def set_image(self, image: np.ndarray | None) -> None:
        super().set_image(image)
        if image is None:
            self._image_widget.clear()
            self._heatmap_widget.clear()
            return
        self._current_widget().set_image(image)

    def _set_submode(self, name: str) -> None:
        self._current_submode = name
        if name == self.SUBMODE_HEATMAP:
            self._view_stack.setCurrentWidget(self._heatmap_widget)
        else:
            self._view_stack.setCurrentWidget(self._image_widget)
        self._update_controls_visibility()
        if self._last_image is not None:
            self._current_widget().set_image(self._last_image)

    def _current_widget(self) -> _ImageMode | _HeatmapMode:
        widget = self._view_stack.currentWidget()
        assert isinstance(widget, _ImageMode | _HeatmapMode)
        return widget

    def _on_colormap_changed(self, friendly_name: str) -> None:
        pg_name = HEATMAP_COLORMAPS.get(friendly_name)
        if pg_name is None:
            return
        self._heatmap_widget.set_colormap(pg_name)

    def _update_controls_visibility(self) -> None:
        is_heatmap = self._current_submode == self.SUBMODE_HEATMAP
        self._colormap_label.setVisible(is_heatmap)
        self._colormap_selector.setVisible(is_heatmap)


class Viz3DPage(_VizPageBase):
    """Activity-bar 3D page — Surface or Point Cloud sub-modes."""

    SUBMODE_SURFACE = "Surface"
    SUBMODE_POINT_CLOUD = "Point Cloud"

    _Z_SLIDER_MIN = 1
    _Z_SLIDER_MAX = 200
    _Z_SLIDER_DEFAULT = 100  # 1.0× in slider units (range 0.01-2.0×)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._surface_widget = _SurfaceMode(self)
        self._point_cloud_widget = _PointCloudMode(self)
        self._view_stack.addWidget(self._surface_widget)
        self._view_stack.addWidget(self._point_cloud_widget)

        btn_surface = self._add_submode(self.SUBMODE_SURFACE, checked=True)
        btn_cloud = self._add_submode(self.SUBMODE_POINT_CLOUD)
        btn_surface.clicked.connect(lambda: self._set_submode(self.SUBMODE_SURFACE))
        btn_cloud.clicked.connect(lambda: self._set_submode(self.SUBMODE_POINT_CLOUD))

        # Right rail: colormap + Z scale.
        self._colormap_label = QLabel("Colormap:", self)
        self._colormap_selector = QComboBox(self)
        for friendly in HEATMAP_COLORMAPS:
            self._colormap_selector.addItem(friendly)
        self._colormap_selector.setCurrentText(DEFAULT_HEATMAP)
        self._colormap_selector.currentTextChanged.connect(self._on_colormap_changed)
        self._rail_layout.addWidget(self._colormap_label)
        self._rail_layout.addWidget(self._colormap_selector)

        self._z_label = QLabel("Z×:", self)
        self._z_slider = QSlider(Qt.Orientation.Horizontal, self)
        self._z_slider.setRange(self._Z_SLIDER_MIN, self._Z_SLIDER_MAX)
        self._z_slider.setValue(self._Z_SLIDER_DEFAULT)
        self._z_slider.valueChanged.connect(self._on_z_changed)
        self._rail_layout.addWidget(self._z_label)
        self._rail_layout.addWidget(self._z_slider)

        self._current_submode = self.SUBMODE_SURFACE
        self._install_param_slot()

    def set_image(self, image: np.ndarray | None) -> None:
        super().set_image(image)
        if image is None:
            self._surface_widget.clear()
            self._point_cloud_widget.clear()
            return
        self._current_widget().set_image(image)

    def _set_submode(self, name: str) -> None:
        self._current_submode = name
        if name == self.SUBMODE_POINT_CLOUD:
            self._view_stack.setCurrentWidget(self._point_cloud_widget)
        else:
            self._view_stack.setCurrentWidget(self._surface_widget)
        if self._last_image is not None:
            self._current_widget().set_image(self._last_image)

    def _current_widget(self) -> _SurfaceMode | _PointCloudMode:
        widget = self._view_stack.currentWidget()
        assert isinstance(widget, _SurfaceMode | _PointCloudMode)
        return widget

    def _on_colormap_changed(self, friendly_name: str) -> None:
        pg_name = HEATMAP_COLORMAPS.get(friendly_name)
        if pg_name is None:
            return
        self._surface_widget.set_colormap(pg_name)
        self._point_cloud_widget.set_colormap(pg_name)

    def _on_z_changed(self, slider_value: int) -> None:
        scale = slider_value / self._Z_SLIDER_DEFAULT
        self._surface_widget.set_z_scale(scale)
        self._point_cloud_widget.set_z_scale(scale)
