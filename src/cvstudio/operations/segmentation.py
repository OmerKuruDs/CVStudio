"""Segmentation & topology operations.

* Distance Transform — pixel-to-foreground distance heat map.
* Connected Components — label each blob, render in a random palette.
* Watershed — full marker-based segmentation, boundaries drawn in red.
* GrabCut — rectangular-init foreground extraction.

All ops self-contain their pipeline so they work straight off a regular image.
Chain a threshold upstream if you want finer control over the binary mask
that feeds Distance Transform / Connected Components.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from cvsandbox.core.operation import OperationSpec, Parameter


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def _to_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def _to_gray_line(in_var: str, out_var: str) -> str:
    return (
        f"{out_var} = cv2.cvtColor({in_var}, cv2.COLOR_BGR2GRAY) "
        f"if {in_var}.ndim == 3 else {in_var}"
    )


def _to_bgr_line(in_var: str, out_var: str) -> str:
    return (
        f"{out_var} = cv2.cvtColor({in_var}, cv2.COLOR_GRAY2BGR) "
        f"if {in_var}.ndim == 2 else {in_var}.copy()"
    )


def _binary_mask(gray: np.ndarray) -> np.ndarray:
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    return mask


# ----------------------------------------------------------- Distance Transform


_DISTANCE_TYPES = {
    "L2 (Euclidean)": cv2.DIST_L2,
    "L1 (Manhattan)": cv2.DIST_L1,
    "Chessboard": cv2.DIST_C,
}
_DISTANCE_CONSTS = {
    "L2 (Euclidean)": "cv2.DIST_L2",
    "L1 (Manhattan)": "cv2.DIST_L1",
    "Chessboard": "cv2.DIST_C",
}

_MASK_SIZES = {"3": 3, "5": 5, "Precise": cv2.DIST_MASK_PRECISE}
_MASK_CONSTS = {"3": "3", "5": "5", "Precise": "cv2.DIST_MASK_PRECISE"}


def _distance_transform(
    image: np.ndarray, distance_type: str, mask_size: str, color_map: bool
) -> np.ndarray:
    gray = _to_gray(image)
    mask = _binary_mask(gray)
    dist = cv2.distanceTransform(mask, _DISTANCE_TYPES[distance_type], _MASK_SIZES[mask_size])
    norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if color_map:
        return cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    return norm


def _distance_transform_code(
    params: dict[str, Any], input_vars: tuple[str, ...], output_var: str
) -> list[str]:
    (a,) = input_vars
    dt = _DISTANCE_CONSTS[params["distance_type"]]
    ms = _MASK_CONSTS[params["mask_size"]]
    use_color = bool(params["color_map"])
    return [
        _to_gray_line(a, "_gray"),
        "_, _mask = cv2.threshold(_gray, 0, 255, cv2.THRESH_BINARY)",
        f"_dist = cv2.distanceTransform(_mask, {dt}, {ms})",
        "_norm = cv2.normalize(_dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)",
        f"{output_var} = cv2.applyColorMap(_norm, cv2.COLORMAP_JET)" if use_color else f"{output_var} = _norm",
    ]


DISTANCE_TRANSFORM = OperationSpec(
    id="segmentation.distance_transform",
    name="Distance Transform",
    category="Segmentation",
    description=(
        "Per-pixel distance to the nearest zero pixel, normalised to 0-255. "
        "Pair with a threshold upstream for a clean mask. Enable <i>color "
        "map</i> for a jet-style heatmap output."
    ),
    parameters=(
        Parameter(
            name="distance_type",
            kind="choice",
            default="L2 (Euclidean)",
            choices=tuple(_DISTANCE_TYPES.keys()),
            label="Distance type",
        ),
        Parameter(
            name="mask_size",
            kind="choice",
            default="3",
            choices=tuple(_MASK_SIZES.keys()),
            label="Mask size",
        ),
        Parameter(name="color_map", kind="bool", default=True, label="Color map (jet)"),
    ),
    func=_distance_transform,
    code_export=_distance_transform_code,
)


# ----------------------------------------------------------- Connected Components


def _component_palette(num_labels: int) -> np.ndarray:
    """Deterministic colour palette: HSV hue sweep → BGR uint8 table."""
    palette = np.zeros((num_labels, 3), dtype=np.uint8)
    if num_labels <= 1:
        return palette
    hues = np.linspace(0, 179, num_labels - 1, dtype=np.uint8)
    hsv = np.stack(
        [hues, np.full_like(hues, 255), np.full_like(hues, 255)], axis=-1
    ).reshape(-1, 1, 3)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).reshape(-1, 3)
    palette[1:] = rgb  # label 0 stays black (background)
    return palette


def _connected_components(
    image: np.ndarray, connectivity: str, min_area: int
) -> np.ndarray:
    gray = _to_gray(image)
    mask = _binary_mask(gray)
    conn = 4 if connectivity == "4-way" else 8
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, conn)
    if int(min_area) > 0:
        for label_id in range(1, num_labels):
            if stats[label_id, cv2.CC_STAT_AREA] < int(min_area):
                labels[labels == label_id] = 0
    palette = _component_palette(num_labels)
    return palette[labels]


def _connected_components_code(
    params: dict[str, Any], input_vars: tuple[str, ...], output_var: str
) -> list[str]:
    (a,) = input_vars
    conn = 4 if params["connectivity"] == "4-way" else 8
    min_area = int(params["min_area"])
    lines = [
        _to_gray_line(a, "_gray"),
        "_, _mask = cv2.threshold(_gray, 0, 255, cv2.THRESH_BINARY)",
        f"_num_labels, _labels, _stats, _ = cv2.connectedComponentsWithStats(_mask, {conn})",
    ]
    if min_area > 0:
        lines.extend(
            [
                f"for _lid in range(1, _num_labels):",
                f"    if _stats[_lid, cv2.CC_STAT_AREA] < {min_area}:",
                f"        _labels[_labels == _lid] = 0",
            ]
        )
    lines.extend(
        [
            "_palette = np.zeros((_num_labels, 3), dtype=np.uint8)",
            "if _num_labels > 1:",
            "    _hues = np.linspace(0, 179, _num_labels - 1, dtype=np.uint8)",
            "    _hsv = np.stack([_hues, np.full_like(_hues, 255), "
            "np.full_like(_hues, 255)], axis=-1).reshape(-1, 1, 3)",
            "    _palette[1:] = cv2.cvtColor(_hsv, cv2.COLOR_HSV2BGR).reshape(-1, 3)",
            f"{output_var} = _palette[_labels]",
        ]
    )
    return lines


CONNECTED_COMPONENTS = OperationSpec(
    id="segmentation.connected_components",
    name="Connected Components",
    category="Segmentation",
    description=(
        "Labels each connected blob in a binary input and paints it a unique "
        "colour. Use <i>min area</i> to drop noise specks."
    ),
    parameters=(
        Parameter(
            name="connectivity",
            kind="choice",
            default="8-way",
            choices=("4-way", "8-way"),
            label="Connectivity",
        ),
        Parameter(
            name="min_area",
            kind="int",
            default=0,
            min=0,
            max=10000,
            label="Min area (px)",
            description="Discard labels smaller than this. 0 = keep all.",
        ),
    ),
    func=_connected_components,
    code_export=_connected_components_code,
)


# ---------------------------------------------------------------------- Watershed


def _watershed(
    image: np.ndarray,
    foreground_threshold: float,
    bg_dilate_iters: int,
    noise_kernel: int,
) -> np.ndarray:
    bgr = _to_bgr(image).copy()
    gray = _to_gray(image)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = max(1, int(noise_kernel) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opened, kernel, iterations=max(1, int(bg_dilate_iters)))
    dist = cv2.distanceTransform(opened, cv2.DIST_L2, 5)
    if dist.max() > 0:
        _, sure_fg = cv2.threshold(
            dist, float(foreground_threshold) * dist.max(), 255, cv2.THRESH_BINARY
        )
    else:
        sure_fg = np.zeros_like(opened)
    sure_fg = sure_fg.astype(np.uint8)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(bgr, markers)
    bgr[markers == -1] = (0, 0, 255)  # boundaries in red
    return bgr


def _watershed_code(
    params: dict[str, Any], input_vars: tuple[str, ...], output_var: str
) -> list[str]:
    (a,) = input_vars
    fg_thr = float(params["foreground_threshold"])
    bg_it = max(1, int(params["bg_dilate_iters"]))
    k = max(1, int(params["noise_kernel"]) | 1)
    return [
        _to_bgr_line(a, output_var),
        _to_gray_line(a, "_gray"),
        "_, _binary = cv2.threshold(_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)",
        f"_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ({k}, {k}))",
        "_opened = cv2.morphologyEx(_binary, cv2.MORPH_OPEN, _kernel, iterations=2)",
        f"_sure_bg = cv2.dilate(_opened, _kernel, iterations={bg_it})",
        "_dist = cv2.distanceTransform(_opened, cv2.DIST_L2, 5)",
        "if _dist.max() > 0:",
        f"    _, _sure_fg = cv2.threshold(_dist, {fg_thr} * _dist.max(), 255, cv2.THRESH_BINARY)",
        "else:",
        "    _sure_fg = np.zeros_like(_opened)",
        "_sure_fg = _sure_fg.astype(np.uint8)",
        "_unknown = cv2.subtract(_sure_bg, _sure_fg)",
        "_, _markers = cv2.connectedComponents(_sure_fg)",
        "_markers = _markers + 1",
        "_markers[_unknown == 255] = 0",
        f"_markers = cv2.watershed({output_var}, _markers)",
        f"{output_var}[_markers == -1] = (0, 0, 255)",
    ]


WATERSHED = OperationSpec(
    id="segmentation.watershed",
    name="Watershed",
    category="Segmentation",
    description=(
        "Marker-based watershed segmentation. Internally: Otsu threshold → "
        "morphological opening → distance transform → connected components → "
        "watershed. Boundaries are drawn in red on the BGR input."
    ),
    parameters=(
        Parameter(
            name="foreground_threshold",
            kind="float",
            default=0.5,
            min=0.05,
            max=0.95,
            step=0.05,
            label="Foreground threshold",
            description="Fraction of the peak distance counted as sure-foreground.",
        ),
        Parameter(
            name="bg_dilate_iters",
            kind="int",
            default=3,
            min=1,
            max=15,
            label="Background dilation",
        ),
        Parameter(
            name="noise_kernel",
            kind="kernel_size",
            default=3,
            min=1,
            max=15,
            step=2,
            label="Noise kernel",
            description="Opening kernel size to clean up small specks.",
        ),
    ),
    func=_watershed,
    code_export=_watershed_code,
)


# ----------------------------------------------------------------------- GrabCut


def _grabcut(image: np.ndarray, margin_pct: int, iterations: int) -> np.ndarray:
    bgr = _to_bgr(image)
    h, w = bgr.shape[:2]
    margin = max(0, int(margin_pct)) / 100.0
    x = int(round(w * margin))
    y = int(round(h * margin))
    rw = max(1, w - 2 * x)
    rh = max(1, h - 2 * y)
    rect = (x, y, rw, rh)
    mask = np.zeros(bgr.shape[:2], dtype=np.uint8)
    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)
    cv2.grabCut(
        bgr,
        mask,
        rect,
        bgd_model,
        fgd_model,
        max(1, int(iterations)),
        cv2.GC_INIT_WITH_RECT,
    )
    fg_mask = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)
    return cv2.bitwise_and(bgr, bgr, mask=fg_mask)


def _grabcut_code(
    params: dict[str, Any], input_vars: tuple[str, ...], output_var: str
) -> list[str]:
    (a,) = input_vars
    margin = max(0, int(params["margin_pct"]))
    iters = max(1, int(params["iterations"]))
    return [
        _to_bgr_line(a, "_bgr"),
        "_h, _w = _bgr.shape[:2]",
        f"_margin = {margin} / 100.0",
        "_x = int(round(_w * _margin))",
        "_y = int(round(_h * _margin))",
        "_rw = max(1, _w - 2 * _x)",
        "_rh = max(1, _h - 2 * _y)",
        "_mask = np.zeros(_bgr.shape[:2], dtype=np.uint8)",
        "_bgd_model = np.zeros((1, 65), dtype=np.float64)",
        "_fgd_model = np.zeros((1, 65), dtype=np.float64)",
        f"cv2.grabCut(_bgr, _mask, (_x, _y, _rw, _rh), _bgd_model, _fgd_model, "
        f"{iters}, cv2.GC_INIT_WITH_RECT)",
        "_fg_mask = np.where((_mask == cv2.GC_FGD) | (_mask == cv2.GC_PR_FGD), "
        "255, 0).astype(np.uint8)",
        f"{output_var} = cv2.bitwise_and(_bgr, _bgr, mask=_fg_mask)",
    ]


GRABCUT = OperationSpec(
    id="segmentation.grabcut",
    name="GrabCut (rect)",
    category="Segmentation",
    description=(
        "Iterative foreground extraction. Initialises with a centred rectangle "
        "(controlled by <i>margin</i>) and refines for <i>iterations</i> passes. "
        "Output: input image with background pixels blacked out."
    ),
    parameters=(
        Parameter(
            name="margin_pct",
            kind="int",
            default=10,
            min=0,
            max=40,
            label="Rect margin (%)",
            description="Percentage of width/height treated as background border.",
        ),
        Parameter(
            name="iterations",
            kind="int",
            default=3,
            min=1,
            max=10,
            label="Iterations",
            description="GrabCut refinement passes. More = slower, possibly cleaner.",
        ),
    ),
    func=_grabcut,
    code_export=_grabcut_code,
)


ALL: tuple[OperationSpec, ...] = (
    DISTANCE_TRANSFORM,
    CONNECTED_COMPONENTS,
    WATERSHED,
    GRABCUT,
)
