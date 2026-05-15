"""Stereo disparity operations — building blocks for 3D visualisation.

Both ops take two inputs (<code>left</code>, <code>right</code>), expect a
rectified pair, and produce a single-channel uint8 disparity map normalised
to 0-255 (brighter = closer). Wire two image sources into a Stereo BM /
SGBM node to get depth-cued grayscale, which a future 3D-viewer step can
lift into a point cloud.
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


def _to_gray_line(in_var: str, out_var: str) -> str:
    return (
        f"{out_var} = cv2.cvtColor({in_var}, cv2.COLOR_BGR2GRAY) "
        f"if {in_var}.ndim == 3 else {in_var}"
    )


def _match_right_to_left(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Make sure right has the same H/W as left so cv2 doesn't blow up."""
    if right.shape[:2] != left.shape[:2]:
        right = cv2.resize(
            right, (left.shape[1], left.shape[0]), interpolation=cv2.INTER_LINEAR
        )
    return right


def _normalise(disp: np.ndarray) -> np.ndarray:
    return cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def _round_disparities(n: int) -> int:
    """numDisparities must be > 0 and divisible by 16."""
    rounded = max(16, (int(n) // 16) * 16)
    return rounded


# ---------------------------------------------------------------- Stereo BM


def _stereo_bm(
    left: np.ndarray,
    right: np.ndarray,
    num_disparities: int,
    block_size: int,
) -> np.ndarray:
    gl = _to_gray(left)
    gr = _to_gray(_match_right_to_left(left, right))
    nd = _round_disparities(num_disparities)
    bs = max(5, int(block_size) | 1)
    matcher = cv2.StereoBM_create(numDisparities=nd, blockSize=bs)
    disp = matcher.compute(gl, gr).astype(np.float32) / 16.0
    return _normalise(disp)


def _stereo_bm_code(
    params: dict[str, Any], input_vars: tuple[str, ...], output_var: str
) -> list[str]:
    a, b = input_vars
    nd = _round_disparities(int(params["num_disparities"]))
    bs = max(5, int(params["block_size"]) | 1)
    return [
        _to_gray_line(a, "_gl"),
        f"_b = {b}",
        f"if _b.shape[:2] != {a}.shape[:2]:",
        f"    _b = cv2.resize(_b, ({a}.shape[1], {a}.shape[0]), interpolation=cv2.INTER_LINEAR)",
        _to_gray_line("_b", "_gr"),
        f"_matcher = cv2.StereoBM_create(numDisparities={nd}, blockSize={bs})",
        "_disp = _matcher.compute(_gl, _gr).astype(np.float32) / 16.0",
        f"{output_var} = cv2.normalize(_disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)",
    ]


STEREO_BM = OperationSpec(
    id="stereo.bm",
    name="Stereo BM (disparity)",
    category="Stereo",
    description=(
        "Block-matching stereo disparity. Fast, classic, but noisy on "
        "textureless regions. Wire the rectified <i>left</i> view from your "
        "chain and the <i>right</i> view from a parallel branch."
    ),
    parameters=(
        Parameter(
            name="num_disparities",
            kind="int",
            default=64,
            min=16,
            max=256,
            step=16,
            label="Num disparities",
            description="Range of disparities searched. Rounded down to a multiple of 16.",
        ),
        Parameter(
            name="block_size",
            kind="kernel_size",
            default=15,
            min=5,
            max=51,
            step=2,
            label="Block size",
            description="Odd, ≥5. Bigger = smoother but less accurate.",
        ),
    ),
    func=_stereo_bm,
    code_export=_stereo_bm_code,
    input_ports=("left", "right"),
)


# -------------------------------------------------------------- Stereo SGBM


def _stereo_sgbm(
    left: np.ndarray,
    right: np.ndarray,
    num_disparities: int,
    block_size: int,
    min_disparity: int,
    uniqueness_ratio: int,
    speckle_window_size: int,
) -> np.ndarray:
    gl = _to_gray(left)
    gr = _to_gray(_match_right_to_left(left, right))
    nd = _round_disparities(num_disparities)
    bs = max(3, int(block_size) | 1)
    matcher = cv2.StereoSGBM_create(
        minDisparity=int(min_disparity),
        numDisparities=nd,
        blockSize=bs,
        P1=8 * bs * bs,
        P2=32 * bs * bs,
        uniquenessRatio=max(0, int(uniqueness_ratio)),
        speckleWindowSize=max(0, int(speckle_window_size)),
        speckleRange=32,
    )
    disp = matcher.compute(gl, gr).astype(np.float32) / 16.0
    return _normalise(disp)


def _stereo_sgbm_code(
    params: dict[str, Any], input_vars: tuple[str, ...], output_var: str
) -> list[str]:
    a, b = input_vars
    nd = _round_disparities(int(params["num_disparities"]))
    bs = max(3, int(params["block_size"]) | 1)
    md = int(params["min_disparity"])
    ur = max(0, int(params["uniqueness_ratio"]))
    sws = max(0, int(params["speckle_window_size"]))
    return [
        _to_gray_line(a, "_gl"),
        f"_b = {b}",
        f"if _b.shape[:2] != {a}.shape[:2]:",
        f"    _b = cv2.resize(_b, ({a}.shape[1], {a}.shape[0]), interpolation=cv2.INTER_LINEAR)",
        _to_gray_line("_b", "_gr"),
        f"_bs = {bs}",
        f"_matcher = cv2.StereoSGBM_create(minDisparity={md}, numDisparities={nd}, "
        f"blockSize=_bs, P1=8 * _bs * _bs, P2=32 * _bs * _bs, "
        f"uniquenessRatio={ur}, speckleWindowSize={sws}, speckleRange=32)",
        "_disp = _matcher.compute(_gl, _gr).astype(np.float32) / 16.0",
        f"{output_var} = cv2.normalize(_disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)",
    ]


STEREO_SGBM = OperationSpec(
    id="stereo.sgbm",
    name="Stereo SGBM (disparity)",
    category="Stereo",
    description=(
        "Semi-global block matching. Slower than plain BM but produces "
        "smoother, more accurate disparity — preferred for any pipeline that "
        "feeds depth into 3D reconstruction."
    ),
    parameters=(
        Parameter(
            name="num_disparities",
            kind="int",
            default=64,
            min=16,
            max=256,
            step=16,
            label="Num disparities",
        ),
        Parameter(
            name="block_size",
            kind="kernel_size",
            default=5,
            min=3,
            max=21,
            step=2,
            label="Block size",
            description="Odd, ≥3. SGBM prefers smaller blocks than BM.",
        ),
        Parameter(
            name="min_disparity",
            kind="int",
            default=0,
            min=-64,
            max=64,
            label="Min disparity",
        ),
        Parameter(
            name="uniqueness_ratio",
            kind="int",
            default=10,
            min=0,
            max=50,
            label="Uniqueness ratio",
            description="Margin (%) by which the best match must beat the runner-up.",
        ),
        Parameter(
            name="speckle_window_size",
            kind="int",
            default=100,
            min=0,
            max=400,
            label="Speckle window",
            description="0 disables speckle filtering. 50-200 typical.",
        ),
    ),
    func=_stereo_sgbm,
    code_export=_stereo_sgbm_code,
    input_ports=("left", "right"),
)


ALL: tuple[OperationSpec, ...] = (STEREO_BM, STEREO_SGBM)
