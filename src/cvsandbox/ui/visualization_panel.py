"""VisualizationPanel — pyqtgraph-backed views of the current pipeline output.

The panel is hosted in a dock widget on the right side of the main window.
It receives every fresh result from PipelineWorker (same signal that updates
ImageView and HistogramPanel) and renders it through one of several modes.

Modes shipped so far:

* <b>2D Image</b> — straight pyqtgraph ImageView, BGR auto-converted to RGB.
  Use for verifying the pipeline output exactly as the main view shows it.
* <b>2D Heatmap</b> — coerces input to single-channel and renders with a
  scientific colormap (viridis, inferno, …). Designed for FFT magnitude,
  distance transforms, disparity maps, or any other intensity field whose
  structure is easier to read in colour than in grayscale.
* <b>3D Surface</b> — lifts the single-channel intensity into a 3-D mesh
  via pyqtgraph.opengl: each pixel becomes a vertex at (x, y, intensity)
  coloured by the same colormap selector as Heatmap. Auto-downsamples
  large inputs so rotation stays smooth.
* <b>3D Point Cloud</b> — same intensity-as-Z mapping but as discrete
  scatter points instead of a triangle mesh. Designed for stereo
  disparity outputs (Stereo BM / SGBM) where holes / speckle look more
  natural as missing points than as bridged mesh triangles. Auto-strides
  the image so the cloud lands near a target point budget for fluid
  rotation.
"""

from __future__ import annotations

import cv2
import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSlider,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Qt

# pyqtgraph defaults to white-on-black plots which clash with the dark UI.
# Force a dark background and light foreground globally so the embedded
# views match the rest of the app.
pg.setConfigOption("background", "#1e1e1e")
pg.setConfigOption("foreground", "#dddddd")
pg.setConfigOption("imageAxisOrder", "row-major")  # match NumPy conventions

# Friendly heatmap labels → the colormap names pyqtgraph ships out of the box.
# We deliberately keep this short — pyqtgraph exposes 70+ CET maps but most
# users only ever want a perceptually uniform sequential (viridis/inferno),
# a classic-looking jet replacement (turbo), a diverging map for phase
# data, or a plain grayscale ramp.
HEATMAP_COLORMAPS: dict[str, str] = {
    "Viridis": "viridis",
    "Inferno": "inferno",
    "Magma": "magma",
    "Plasma": "plasma",
    "Turbo (jet-like)": "turbo",
    "Grayscale": "CET-L01",
    "Diverging (phase)": "CET-D01",
}
DEFAULT_HEATMAP = "Viridis"

# Performance cap: the surface plot stays interactive up to ~256x256 vertices
# (~64k tris). Above that, downsample so rotation/zoom stays smooth.
SURFACE_MAX_GRID = 192

# Target point count for 3D Point Cloud mode. We pick an integer stride that
# brings the per-pixel grid down to roughly this many points so a 1080p
# disparity map doesn't melt the renderer.
POINT_CLOUD_TARGET_POINTS = 30_000


def _bgr_to_rgb_or_gray(image: np.ndarray) -> np.ndarray:
    """Convert OpenCV-style BGR to RGB so colour channels display correctly.

    Grayscale and single-channel inputs are passed through untouched.
    """
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    return image


def _coerce_to_single_channel(image: np.ndarray) -> np.ndarray:
    """Reduce any input to a 2-D array suitable for heatmap rendering.

    BGR / BGRA images are converted to grayscale via the standard luminance
    weights — the same path the heatmap viewer would take anyway, but done
    by OpenCV which is faster and consistent with the rest of the pipeline.
    """
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return image


def _downsample_for_surface(image: np.ndarray, max_dim: int = SURFACE_MAX_GRID) -> np.ndarray:
    """Shrink a 2-D image so its longest side is at most ``max_dim`` pixels.

    Returns the input unchanged if it is already at or below the cap. Uses
    ``INTER_AREA`` which is the highest-quality downsampling interpolation
    OpenCV offers.
    """
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return image
    scale = max_dim / longest
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _heights_to_rgba(values: np.ndarray, colormap_name: str) -> np.ndarray:
    """Map a 2-D float array to a (..., 4) float32 RGBA grid via colormap."""
    try:
        cmap = pg.colormap.get(colormap_name)
    except (FileNotFoundError, ValueError):
        cmap = pg.colormap.get("viridis")
    rng = float(values.max()) - float(values.min())
    if rng <= 0:
        normalised = np.zeros_like(values, dtype=np.float32)
    else:
        normalised = (values.astype(np.float32) - float(values.min())) / rng
    rgba = cmap.map(normalised, mode=pg.ColorMap.FLOAT)
    return rgba.astype(np.float32)


def _auto_stride(h: int, w: int, target_points: int = POINT_CLOUD_TARGET_POINTS) -> int:
    """Pick the smallest integer stride that keeps point count below the target."""
    if h * w <= target_points:
        return 1
    return max(1, int(np.ceil(np.sqrt(h * w / target_points))))


class _ImageMode(QWidget):
    """A pyqtgraph ImageView wrapper — the 2D-image rendering mode."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._view = pg.ImageView(parent=self)
        # The default ImageView has a histogram + LUT control on the right
        # and timeline controls below. We're only displaying still images
        # for now, so hide the timeline. The histogram stays because it
        # also doubles as a colour-map control for single-channel images.
        self._view.ui.roiBtn.setVisible(False)
        self._view.ui.menuBtn.setVisible(False)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._view)

    def set_image(self, image: np.ndarray) -> None:
        # pyqtgraph keeps the previous view box state when you call setImage
        # again, so panning / zoom survives pipeline updates.
        self._view.setImage(
            _bgr_to_rgb_or_gray(image),
            autoLevels=True,
            autoHistogramRange=False,
        )

    def clear(self) -> None:
        self._view.clear()


class _HeatmapMode(QWidget):
    """Single-channel colour-mapped view.

    Reuses pyqtgraph's ImageView because the side-panel histogram is the
    most intuitive colorbar control in the toolkit — drag the yellow
    levels-region to clip dynamic range live. The colormap itself comes
    from the panel toolbar's selector, applied via ``set_colormap``.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._view = pg.ImageView(parent=self)
        self._view.ui.roiBtn.setVisible(False)
        self._view.ui.menuBtn.setVisible(False)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._view)
        self._colormap_name = HEATMAP_COLORMAPS[DEFAULT_HEATMAP]
        self._apply_colormap()

    def set_image(self, image: np.ndarray) -> None:
        gray = _coerce_to_single_channel(image)
        self._view.setImage(
            gray,
            autoLevels=True,
            autoHistogramRange=False,
        )
        # ImageView resets the LUT to grayscale on each new image set, so
        # re-apply the selected colormap whenever new data arrives.
        self._apply_colormap()

    def clear(self) -> None:
        self._view.clear()

    def set_colormap(self, name: str) -> None:
        """Switch active colormap. Accepts any pyqtgraph-known map name."""
        self._colormap_name = name
        self._apply_colormap()

    def _apply_colormap(self) -> None:
        try:
            cmap = pg.colormap.get(self._colormap_name)
        except (FileNotFoundError, ValueError):
            cmap = pg.colormap.get("viridis")
        self._view.setColorMap(cmap)


class _SurfaceMode(QWidget):
    """Intensity-as-Z 3-D surface render.

    A pyqtgraph GLViewWidget hosts a single GLSurfacePlotItem. Each call to
    ``set_image`` downsamples the input (so the surface mesh stays under the
    SURFACE_MAX_GRID cap) and rebuilds the mesh's height + colour grids.

    Use the mouse to rotate (left drag), pan (Shift + drag), and zoom
    (wheel) the camera. The Z-scale slider on the panel toolbar tunes how
    exaggerated the height looks relative to the image extent.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._view = gl.GLViewWidget(parent=self)
        # Move the default camera out far enough to see a 256-wide grid.
        self._view.setCameraPosition(distance=350, elevation=30, azimuth=-60)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._view)
        self._surface: gl.GLSurfacePlotItem | None = None
        self._colormap_name = HEATMAP_COLORMAPS[DEFAULT_HEATMAP]
        self._z_scale = 1.0
        self._last_gray: np.ndarray | None = None

    def set_image(self, image: np.ndarray) -> None:
        gray = _coerce_to_single_channel(image)
        downsampled = _downsample_for_surface(gray)
        self._last_gray = downsampled
        self._rebuild_surface()

    def clear(self) -> None:
        if self._surface is not None:
            self._view.removeItem(self._surface)
            self._surface = None
        self._last_gray = None

    def set_colormap(self, name: str) -> None:
        self._colormap_name = name
        if self._last_gray is not None:
            self._rebuild_surface()

    def set_z_scale(self, scale: float) -> None:
        self._z_scale = max(0.01, float(scale))
        if self._last_gray is not None:
            self._rebuild_surface()

    def _rebuild_surface(self) -> None:
        if self._last_gray is None:
            return
        z_np = self._last_gray.astype(np.float32)  # numpy convention: (H, W)
        h, w = z_np.shape
        # pyqtgraph's GLSurfacePlotItem expects z.shape == (len(x), len(y))
        # i.e. (W, H) — transposed relative to numpy's (rows, cols) layout.
        x = np.linspace(-w / 2, w / 2, w, dtype=np.float32)
        y = np.linspace(-h / 2, h / 2, h, dtype=np.float32)
        # Match Z extent to the longer image axis so the default 1× scale
        # produces a balanced surface for any input size.
        peak = float(z_np.max() - z_np.min()) or 1.0
        target_height = max(w, h) * 0.4
        z_scaled = (z_np - float(z_np.min())) * (target_height / peak) * self._z_scale
        z_for_pg = np.ascontiguousarray(z_scaled.T)
        colors = _heights_to_rgba(z_np, self._colormap_name)
        colors_for_pg = np.ascontiguousarray(colors.transpose(1, 0, 2))
        if self._surface is None:
            self._surface = gl.GLSurfacePlotItem(
                x=x, y=y, z=z_for_pg, colors=colors_for_pg, smooth=False, shader=None
            )
            self._view.addItem(self._surface)
        else:
            self._surface.setData(x=x, y=y, z=z_for_pg, colors=colors_for_pg)


class _PointCloudMode(QWidget):
    """Single-channel input rendered as a 3-D scatter point cloud.

    Designed for stereo disparity outputs: each non-zero pixel becomes a
    coloured dot at (col − cx, −(row − cy), intensity-as-Z). Auto-strides
    the input so the rendered cloud stays near
    :data:`POINT_CLOUD_TARGET_POINTS` regardless of source resolution.

    Unlike Surface mode this draws no triangles, so speckle / holes in a
    disparity map appear as gaps in the cloud instead of stretched mesh
    facets — usually a more honest visualisation of stereo output.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._view = gl.GLViewWidget(parent=self)
        self._view.setCameraPosition(distance=350, elevation=30, azimuth=-60)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._view)
        self._scatter: gl.GLScatterPlotItem | None = None
        self._colormap_name = HEATMAP_COLORMAPS[DEFAULT_HEATMAP]
        self._z_scale = 1.0
        self._last_gray: np.ndarray | None = None

    def set_image(self, image: np.ndarray) -> None:
        self._last_gray = _coerce_to_single_channel(image)
        self._rebuild()

    def clear(self) -> None:
        if self._scatter is not None:
            self._view.removeItem(self._scatter)
            self._scatter = None
        self._last_gray = None

    def set_colormap(self, name: str) -> None:
        self._colormap_name = name
        if self._last_gray is not None:
            self._rebuild()

    def set_z_scale(self, scale: float) -> None:
        self._z_scale = max(0.01, float(scale))
        if self._last_gray is not None:
            self._rebuild()

    def _rebuild(self) -> None:
        if self._last_gray is None:
            return
        gray = self._last_gray
        h, w = gray.shape[:2]
        stride = _auto_stride(h, w)
        # Sample on the strided grid. ys/xs end up as 2-D index grids.
        ys, xs = np.mgrid[0:h:stride, 0:w:stride]
        zs = gray[::stride, ::stride].astype(np.float32)
        peak = float(zs.max() - zs.min()) or 1.0
        target_height = max(w, h) * 0.4
        z_scaled = (zs - float(zs.min())) * (target_height / peak) * self._z_scale
        # Centre on origin; flip Y so the cloud's "top" matches the image's top.
        pts = np.stack(
            [
                xs.ravel().astype(np.float32) - w / 2.0,
                -(ys.ravel().astype(np.float32) - h / 2.0),
                z_scaled.ravel(),
            ],
            axis=-1,
        )
        colors = _heights_to_rgba(zs, self._colormap_name).reshape(-1, 4)
        if self._scatter is None:
            self._scatter = gl.GLScatterPlotItem(
                pos=pts, color=colors, size=3.0, pxMode=True
            )
            self._view.addItem(self._scatter)
        else:
            self._scatter.setData(pos=pts, color=colors, size=3.0)


class VisualizationPanel(QWidget):
    """The dock-hosted viz panel.

    The widget exposes ``set_image`` / ``clear`` matching the convention used
    by the existing ImageView and HistogramPanel, so MainWindow can wire it
    in next to those without special-casing.
    """

    MODE_2D_IMAGE = "2D Image"
    MODE_2D_HEATMAP = "2D Heatmap"
    MODE_3D_SURFACE = "3D Surface"
    MODE_3D_POINT_CLOUD = "3D Point Cloud"

    _Z_SLIDER_MIN = 1
    _Z_SLIDER_MAX = 200
    _Z_SLIDER_DEFAULT = 100  # represents 1.0× in slider units (range = 0.01-2.0×)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("VisualizationPanel")
        self.setMinimumWidth(280)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._mode_selector = QComboBox(self)
        self._mode_selector.addItem(self.MODE_2D_IMAGE)
        self._mode_selector.addItem(self.MODE_2D_HEATMAP)
        self._mode_selector.addItem(self.MODE_3D_SURFACE)
        self._mode_selector.addItem(self.MODE_3D_POINT_CLOUD)
        self._mode_selector.currentTextChanged.connect(self._on_mode_changed)

        self._colormap_label = QLabel("Colormap:", self)
        self._colormap_selector = QComboBox(self)
        for friendly in HEATMAP_COLORMAPS:
            self._colormap_selector.addItem(friendly)
        self._colormap_selector.setCurrentText(DEFAULT_HEATMAP)
        self._colormap_selector.currentTextChanged.connect(self._on_colormap_changed)

        self._z_label = QLabel("Z×:", self)
        self._z_slider = QSlider(Qt.Orientation.Horizontal, self)
        self._z_slider.setRange(self._Z_SLIDER_MIN, self._Z_SLIDER_MAX)
        self._z_slider.setValue(self._Z_SLIDER_DEFAULT)
        self._z_slider.setMinimumWidth(80)
        self._z_slider.setMaximumWidth(140)
        self._z_slider.valueChanged.connect(self._on_z_scale_changed)

        toolbar_row = QHBoxLayout()
        toolbar_row.setContentsMargins(6, 4, 6, 4)
        toolbar_row.addWidget(QLabel("Mode:", self))
        toolbar_row.addWidget(self._mode_selector, 1)
        toolbar_row.addWidget(self._colormap_label)
        toolbar_row.addWidget(self._colormap_selector, 1)
        toolbar_row.addWidget(self._z_label)
        toolbar_row.addWidget(self._z_slider)

        self._stack = QStackedWidget(self)
        self._image_mode = _ImageMode(self)
        self._heatmap_mode = _HeatmapMode(self)
        self._surface_mode = _SurfaceMode(self)
        self._point_cloud_mode = _PointCloudMode(self)
        self._stack.addWidget(self._image_mode)  # index 0
        self._stack.addWidget(self._heatmap_mode)  # index 1
        self._stack.addWidget(self._surface_mode)  # index 2
        self._stack.addWidget(self._point_cloud_mode)  # index 3

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addLayout(toolbar_row)
        root.addWidget(self._stack, 1)

        self._last_image: np.ndarray | None = None
        self._update_toolbar_visibility(self.MODE_2D_IMAGE)

    # ------------------------------------------------------------------ public

    def set_image(self, image: np.ndarray | None) -> None:
        self._last_image = image
        if image is None:
            self._image_mode.clear()
            self._heatmap_mode.clear()
            self._surface_mode.clear()
            self._point_cloud_mode.clear()
            return
        self._current_mode_widget().set_image(image)

    def clear(self) -> None:
        self.set_image(None)

    # ------------------------------------------------------------------ internal

    def _current_mode_widget(
        self,
    ) -> _ImageMode | _HeatmapMode | _SurfaceMode | _PointCloudMode:
        widget = self._stack.currentWidget()
        assert isinstance(
            widget, _ImageMode | _HeatmapMode | _SurfaceMode | _PointCloudMode
        )
        return widget

    def _on_mode_changed(self, name: str) -> None:
        if name == self.MODE_2D_HEATMAP:
            self._stack.setCurrentWidget(self._heatmap_mode)
        elif name == self.MODE_3D_SURFACE:
            self._stack.setCurrentWidget(self._surface_mode)
        elif name == self.MODE_3D_POINT_CLOUD:
            self._stack.setCurrentWidget(self._point_cloud_mode)
        else:
            self._stack.setCurrentWidget(self._image_mode)
        self._update_toolbar_visibility(name)
        if self._last_image is not None:
            self._current_mode_widget().set_image(self._last_image)

    def _on_colormap_changed(self, friendly_name: str) -> None:
        pg_name = HEATMAP_COLORMAPS.get(friendly_name)
        if pg_name is None:
            return
        self._heatmap_mode.set_colormap(pg_name)
        self._surface_mode.set_colormap(pg_name)
        self._point_cloud_mode.set_colormap(pg_name)

    def _on_z_scale_changed(self, slider_value: int) -> None:
        # slider 100 → 1.0×; slider 200 → 2.0×; slider 1 → 0.01×
        scale = slider_value / self._Z_SLIDER_DEFAULT
        self._surface_mode.set_z_scale(scale)
        self._point_cloud_mode.set_z_scale(scale)

    def _update_toolbar_visibility(self, mode_name: str) -> None:
        is_heatmap = mode_name == self.MODE_2D_HEATMAP
        is_surface = mode_name == self.MODE_3D_SURFACE
        is_cloud = mode_name == self.MODE_3D_POINT_CLOUD
        colorable = is_heatmap or is_surface or is_cloud
        z_capable = is_surface or is_cloud
        self._colormap_label.setVisible(colorable)
        self._colormap_selector.setVisible(colorable)
        self._z_label.setVisible(z_capable)
        self._z_slider.setVisible(z_capable)
