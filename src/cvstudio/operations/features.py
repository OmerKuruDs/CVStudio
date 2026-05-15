"""Feature detection & Hough operations: corners, keypoints, lines, circles.

These ops detect interest points / parametric shapes and draw them as overlays
on a BGR copy of the input. Output is always a 3-channel BGR image so colored
markings survive subsequent steps in the pipeline. Hough ops apply an internal
Canny pass so they "just work" on photographic inputs; chain a dedicated Canny
upstream if you want more control over the edge map.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from cvstudio.core.operation import OperationSpec, Parameter


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


# ---------------------------------------------------------------- Harris Corners


def _harris(
    image: np.ndarray,
    block_size: int,
    ksize: int,
    k: float,
    threshold: float,
) -> np.ndarray:
    gray = _to_gray(image).astype(np.float32)
    aperture = max(3, min(7, int(ksize) | 1))
    response = cv2.cornerHarris(gray, int(block_size), aperture, float(k))
    canvas = _to_bgr(image).copy()
    peak = float(response.max()) if response.size else 0.0
    if peak > 0:
        mask = response > float(threshold) * peak
        canvas[mask] = (0, 0, 255)  # BGR red
    return canvas


def _harris_code(
    params: dict[str, Any], input_vars: tuple[str, ...], output_var: str
) -> list[str]:
    (a,) = input_vars
    aperture = max(3, min(7, int(params["ksize"]) | 1))
    block = int(params["block_size"])
    k = float(params["k"])
    thr = float(params["threshold"])
    return [
        _to_gray_line(a, "_gray"),
        "_gray = _gray.astype(np.float32)",
        f"_response = cv2.cornerHarris(_gray, {block}, {aperture}, {k})",
        _to_bgr_line(a, output_var),
        "_peak = float(_response.max()) if _response.size else 0.0",
        "if _peak > 0:",
        f"    _mask = _response > {thr} * _peak",
        f"    {output_var}[_mask] = (0, 0, 255)",
    ]


HARRIS = OperationSpec(
    id="features.harris",
    name="Harris Corners",
    category="Features",
    description=(
        "Marks Harris corner responses in red. `threshold` is a fraction of the "
        "peak response — lower values mark more pixels."
    ),
    parameters=(
        Parameter(
            name="block_size",
            kind="int",
            default=2,
            min=2,
            max=10,
            label="Block size",
            description="Neighborhood size for the corner response.",
        ),
        Parameter(
            name="ksize",
            kind="kernel_size",
            default=3,
            min=3,
            max=7,
            step=2,
            label="Sobel aperture",
        ),
        Parameter(
            name="k",
            kind="float",
            default=0.04,
            min=0.01,
            max=0.2,
            step=0.01,
            label="Harris k",
            description="Free parameter — typical range 0.04-0.06.",
        ),
        Parameter(
            name="threshold",
            kind="float",
            default=0.01,
            min=0.001,
            max=0.5,
            step=0.001,
            label="Threshold (×peak)",
            description="Mark pixels whose response exceeds this fraction of the peak.",
        ),
    ),
    func=_harris,
    code_export=_harris_code,
)


# ------------------------------------------------------------- Shi-Tomasi Corners


def _shi_tomasi(
    image: np.ndarray,
    max_corners: int,
    quality_level: float,
    min_distance: int,
    block_size: int,
    radius: int,
) -> np.ndarray:
    gray = _to_gray(image)
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max(1, int(max_corners)),
        qualityLevel=max(0.001, float(quality_level)),
        minDistance=max(1, int(min_distance)),
        blockSize=max(3, int(block_size)),
    )
    canvas = _to_bgr(image).copy()
    if corners is not None:
        for c in corners.reshape(-1, 2):
            x, y = int(round(float(c[0]))), int(round(float(c[1])))
            cv2.circle(canvas, (x, y), int(radius), (0, 255, 0), thickness=1)
    return canvas


def _shi_tomasi_code(
    params: dict[str, Any], input_vars: tuple[str, ...], output_var: str
) -> list[str]:
    (a,) = input_vars
    max_c = max(1, int(params["max_corners"]))
    q = max(0.001, float(params["quality_level"]))
    md = max(1, int(params["min_distance"]))
    bs = max(3, int(params["block_size"]))
    r = int(params["radius"])
    return [
        _to_gray_line(a, "_gray"),
        f"_corners = cv2.goodFeaturesToTrack(_gray, maxCorners={max_c}, "
        f"qualityLevel={q}, minDistance={md}, blockSize={bs})",
        _to_bgr_line(a, output_var),
        "if _corners is not None:",
        "    for _c in _corners.reshape(-1, 2):",
        f"        cv2.circle({output_var}, (int(round(float(_c[0]))), "
        f"int(round(float(_c[1])))), {r}, (0, 255, 0), thickness=1)",
    ]


SHI_TOMASI = OperationSpec(
    id="features.shi_tomasi",
    name="Shi-Tomasi Corners",
    category="Features",
    description=(
        "Draws the N strongest Shi-Tomasi corners as green circles. Better "
        "tracking quality than Harris in most natural images."
    ),
    parameters=(
        Parameter(
            name="max_corners",
            kind="int",
            default=100,
            min=1,
            max=1000,
            label="Max corners",
        ),
        Parameter(
            name="quality_level",
            kind="float",
            default=0.01,
            min=0.001,
            max=1.0,
            step=0.001,
            label="Quality level",
            description="Minimum accepted quality, fraction of the best corner's score.",
        ),
        Parameter(
            name="min_distance",
            kind="int",
            default=10,
            min=1,
            max=200,
            label="Min distance (px)",
        ),
        Parameter(
            name="block_size",
            kind="int",
            default=3,
            min=3,
            max=21,
            label="Block size",
        ),
        Parameter(
            name="radius",
            kind="int",
            default=3,
            min=1,
            max=15,
            label="Marker radius",
        ),
    ),
    func=_shi_tomasi,
    code_export=_shi_tomasi_code,
)


# ----------------------------------------------------------------- FAST Keypoints


def _fast(image: np.ndarray, threshold: int, nonmax: bool) -> np.ndarray:
    gray = _to_gray(image)
    detector = cv2.FastFeatureDetector_create(
        threshold=int(threshold), nonmaxSuppression=bool(nonmax)
    )
    keypoints = detector.detect(gray, None)
    canvas = _to_bgr(image).copy()
    return cv2.drawKeypoints(canvas, keypoints, None, color=(0, 255, 255))


def _fast_code(
    params: dict[str, Any], input_vars: tuple[str, ...], output_var: str
) -> list[str]:
    (a,) = input_vars
    thr = int(params["threshold"])
    nonmax = bool(params["nonmax"])
    return [
        _to_gray_line(a, "_gray"),
        f"_detector = cv2.FastFeatureDetector_create(threshold={thr}, "
        f"nonmaxSuppression={nonmax})",
        "_kps = _detector.detect(_gray, None)",
        _to_bgr_line(a, "_canvas"),
        f"{output_var} = cv2.drawKeypoints(_canvas, _kps, None, color=(0, 255, 255))",
    ]


FAST = OperationSpec(
    id="features.fast",
    name="FAST Keypoints",
    category="Features",
    description=(
        "Detects FAST (Features from Accelerated Segment Test) keypoints and "
        "draws them in yellow. Fast and rotation-invariant; no descriptors."
    ),
    parameters=(
        Parameter(
            name="threshold",
            kind="int",
            default=10,
            min=1,
            max=100,
            label="Threshold",
            description="Intensity difference required for a corner.",
        ),
        Parameter(
            name="nonmax",
            kind="bool",
            default=True,
            label="Non-max suppression",
        ),
    ),
    func=_fast,
    code_export=_fast_code,
)


# ------------------------------------------------------------------ ORB Keypoints


def _orb(
    image: np.ndarray,
    nfeatures: int,
    scale_factor: float,
    nlevels: int,
    rich: bool,
) -> np.ndarray:
    gray = _to_gray(image)
    detector = cv2.ORB_create(
        nfeatures=max(1, int(nfeatures)),
        scaleFactor=max(1.01, float(scale_factor)),
        nlevels=max(1, int(nlevels)),
    )
    keypoints = detector.detect(gray, None)
    canvas = _to_bgr(image).copy()
    flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS if rich else 0
    return cv2.drawKeypoints(canvas, keypoints, None, color=(255, 0, 255), flags=flags)


def _orb_code(
    params: dict[str, Any], input_vars: tuple[str, ...], output_var: str
) -> list[str]:
    (a,) = input_vars
    nf = max(1, int(params["nfeatures"]))
    sf = max(1.01, float(params["scale_factor"]))
    nl = max(1, int(params["nlevels"]))
    rich = bool(params["rich"])
    flags = "cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS" if rich else "0"
    return [
        _to_gray_line(a, "_gray"),
        f"_detector = cv2.ORB_create(nfeatures={nf}, scaleFactor={sf}, nlevels={nl})",
        "_kps = _detector.detect(_gray, None)",
        _to_bgr_line(a, "_canvas"),
        f"{output_var} = cv2.drawKeypoints(_canvas, _kps, None, "
        f"color=(255, 0, 255), flags={flags})",
    ]


ORB = OperationSpec(
    id="features.orb",
    name="ORB Keypoints",
    category="Features",
    description=(
        "Oriented FAST + Rotated BRIEF. Detects scale & rotation invariant "
        "keypoints. Toggle <i>rich</i> to draw size + orientation circles."
    ),
    parameters=(
        Parameter(
            name="nfeatures",
            kind="int",
            default=500,
            min=50,
            max=5000,
            label="Max features",
        ),
        Parameter(
            name="scale_factor",
            kind="float",
            default=1.2,
            min=1.05,
            max=2.0,
            step=0.05,
            label="Scale factor",
        ),
        Parameter(
            name="nlevels",
            kind="int",
            default=8,
            min=1,
            max=16,
            label="Pyramid levels",
        ),
        Parameter(
            name="rich",
            kind="bool",
            default=False,
            label="Draw rich keypoints",
        ),
    ),
    func=_orb,
    code_export=_orb_code,
)


# ------------------------------------------------------------------- Hough Lines


def _hough_lines(
    image: np.ndarray,
    canny_low: int,
    canny_high: int,
    threshold: int,
    min_line_length: int,
    max_line_gap: int,
    thickness: int,
) -> np.ndarray:
    gray = _to_gray(image)
    edges = cv2.Canny(gray, float(canny_low), float(canny_high))
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=max(1, int(threshold)),
        minLineLength=max(0, int(min_line_length)),
        maxLineGap=max(0, int(max_line_gap)),
    )
    canvas = _to_bgr(image).copy()
    if lines is not None:
        for seg in lines.reshape(-1, 4):
            x1, y1, x2, y2 = (int(v) for v in seg)
            cv2.line(canvas, (x1, y1), (x2, y2), (0, 255, 0), int(thickness))
    return canvas


def _hough_lines_code(
    params: dict[str, Any], input_vars: tuple[str, ...], output_var: str
) -> list[str]:
    (a,) = input_vars
    low = float(params["canny_low"])
    high = float(params["canny_high"])
    thr = max(1, int(params["threshold"]))
    mll = max(0, int(params["min_line_length"]))
    mlg = max(0, int(params["max_line_gap"]))
    th = int(params["thickness"])
    return [
        _to_gray_line(a, "_gray"),
        f"_edges = cv2.Canny(_gray, {low}, {high})",
        f"_lines = cv2.HoughLinesP(_edges, rho=1, theta=np.pi / 180, "
        f"threshold={thr}, minLineLength={mll}, maxLineGap={mlg})",
        _to_bgr_line(a, output_var),
        "if _lines is not None:",
        "    for _seg in _lines.reshape(-1, 4):",
        f"        cv2.line({output_var}, (int(_seg[0]), int(_seg[1])), "
        f"(int(_seg[2]), int(_seg[3])), (0, 255, 0), {th})",
    ]


HOUGH_LINES = OperationSpec(
    id="features.hough_lines",
    name="Hough Lines",
    category="Features",
    description=(
        "Probabilistic Hough line transform. Runs Canny internally, then "
        "detects line segments and draws them in green."
    ),
    parameters=(
        Parameter(
            name="canny_low",
            kind="int",
            default=50,
            min=0,
            max=500,
            label="Canny low",
        ),
        Parameter(
            name="canny_high",
            kind="int",
            default=150,
            min=0,
            max=500,
            label="Canny high",
        ),
        Parameter(
            name="threshold",
            kind="int",
            default=80,
            min=1,
            max=500,
            label="Accumulator threshold",
            description="Minimum votes for a line to be detected.",
        ),
        Parameter(
            name="min_line_length",
            kind="int",
            default=50,
            min=0,
            max=1000,
            label="Min line length (px)",
        ),
        Parameter(
            name="max_line_gap",
            kind="int",
            default=10,
            min=0,
            max=500,
            label="Max line gap (px)",
        ),
        Parameter(
            name="thickness",
            kind="int",
            default=2,
            min=1,
            max=10,
            label="Line thickness",
        ),
    ),
    func=_hough_lines,
    code_export=_hough_lines_code,
)


# ----------------------------------------------------------------- Hough Circles


def _hough_circles(
    image: np.ndarray,
    dp: float,
    min_dist: int,
    canny_threshold: int,
    accumulator_threshold: int,
    min_radius: int,
    max_radius: int,
    thickness: int,
) -> np.ndarray:
    gray = _to_gray(image)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=max(1.0, float(dp)),
        minDist=max(1, int(min_dist)),
        param1=max(1, int(canny_threshold)),
        param2=max(1, int(accumulator_threshold)),
        minRadius=max(0, int(min_radius)),
        maxRadius=max(0, int(max_radius)),
    )
    canvas = _to_bgr(image).copy()
    if circles is not None:
        for c in np.round(circles[0]).astype(int):
            cx, cy, r = int(c[0]), int(c[1]), int(c[2])
            cv2.circle(canvas, (cx, cy), r, (0, 255, 0), int(thickness))
            cv2.circle(canvas, (cx, cy), 2, (0, 0, 255), -1)
    return canvas


def _hough_circles_code(
    params: dict[str, Any], input_vars: tuple[str, ...], output_var: str
) -> list[str]:
    (a,) = input_vars
    dp = max(1.0, float(params["dp"]))
    md = max(1, int(params["min_dist"]))
    p1 = max(1, int(params["canny_threshold"]))
    p2 = max(1, int(params["accumulator_threshold"]))
    minr = max(0, int(params["min_radius"]))
    maxr = max(0, int(params["max_radius"]))
    th = int(params["thickness"])
    return [
        _to_gray_line(a, "_gray"),
        f"_circles = cv2.HoughCircles(_gray, cv2.HOUGH_GRADIENT, dp={dp}, "
        f"minDist={md}, param1={p1}, param2={p2}, minRadius={minr}, maxRadius={maxr})",
        _to_bgr_line(a, output_var),
        "if _circles is not None:",
        "    for _c in np.round(_circles[0]).astype(int):",
        f"        cv2.circle({output_var}, (int(_c[0]), int(_c[1])), int(_c[2]), "
        f"(0, 255, 0), {th})",
        f"        cv2.circle({output_var}, (int(_c[0]), int(_c[1])), 2, (0, 0, 255), -1)",
    ]


HOUGH_CIRCLES = OperationSpec(
    id="features.hough_circles",
    name="Hough Circles",
    category="Features",
    description=(
        "Detects circles via Hough gradient method. Draws the rim in green and "
        "the center in red. Tune <i>min/max radius</i> to the expected size."
    ),
    parameters=(
        Parameter(
            name="dp",
            kind="float",
            default=1.5,
            min=1.0,
            max=3.0,
            step=0.1,
            label="dp (inv. resolution)",
            description="1.0 = same resolution as input; 2.0 = half.",
        ),
        Parameter(
            name="min_dist",
            kind="int",
            default=30,
            min=1,
            max=1000,
            label="Min center distance",
        ),
        Parameter(
            name="canny_threshold",
            kind="int",
            default=100,
            min=1,
            max=500,
            label="Canny high threshold",
        ),
        Parameter(
            name="accumulator_threshold",
            kind="int",
            default=30,
            min=1,
            max=300,
            label="Accumulator threshold",
            description="Smaller = more false circles.",
        ),
        Parameter(
            name="min_radius",
            kind="int",
            default=0,
            min=0,
            max=1000,
            label="Min radius (px)",
        ),
        Parameter(
            name="max_radius",
            kind="int",
            default=0,
            min=0,
            max=1000,
            label="Max radius (px)",
            description="0 = OpenCV picks an upper bound.",
        ),
        Parameter(
            name="thickness",
            kind="int",
            default=2,
            min=1,
            max=10,
            label="Line thickness",
        ),
    ),
    func=_hough_circles,
    code_export=_hough_circles_code,
)


ALL: tuple[OperationSpec, ...] = (
    HARRIS,
    SHI_TOMASI,
    FAST,
    ORB,
    HOUGH_LINES,
    HOUGH_CIRCLES,
)
