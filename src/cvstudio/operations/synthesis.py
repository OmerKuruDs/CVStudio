"""Synthesis operations — synthetic defect generation primitives.

Classic-CV building blocks for growing labelled training data from a small
seed set: extract a defect's geometry from one image, transfer it onto a
clean substrate, overlay procedural noise. Three ops here are stochastic
(copy-paste with jitter, perlin overlay) and take a ``seed`` parameter so
the pipeline's "same params → same output" invariant survives — pick a seed
to reproduce a specific variant, vary it to generate a dataset.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from cvstudio.core.operation import OperationSpec, Parameter
from cvstudio.defects.grain_flattening import inject_grain_flattening


# --------------------------------------------------------------- boundary_extract


def _boundary_extract(image: np.ndarray, thickness: int) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    binary = (gray > 127).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = np.zeros_like(binary)
    if contours:
        cv2.drawContours(out, contours, -1, 255, thickness=int(thickness))
    return out


def _boundary_extract_code(
    params: dict[str, Any], input_vars: tuple[str, ...], output_var: str
) -> list[str]:
    (image,) = input_vars
    thickness = int(params["thickness"])
    return [
        f"_gray = cv2.cvtColor({image}, cv2.COLOR_BGR2GRAY) if {image}.ndim == 3 else {image}",
        "_binary = (_gray > 127).astype('uint8') * 255",
        "_contours, _ = cv2.findContours(_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)",
        f"{output_var} = np.zeros_like(_binary)",
        "if _contours:",
        f"    cv2.drawContours({output_var}, _contours, -1, 255, thickness={thickness})",
    ]


BOUNDARY_EXTRACT = OperationSpec(
    id="synthesis.boundary_extract",
    name="Boundary Extract",
    category="Synthesis",
    description=(
        "Extract every external contour of a binarised input as a thin line "
        "mask. Use after a Threshold to isolate the geometry of object edges "
        "(defective boundaries, silhouettes) without their fill."
    ),
    parameters=(
        Parameter(
            name="thickness", kind="int", default=1, min=1, max=20, step=1,
            label="Line thickness",
            description="Pixel width of the drawn contour line.",
        ),
    ),
    func=_boundary_extract,
    code_export=_boundary_extract_code,
)


# --------------------------------------------------------------- boundary_smooth


def _boundary_smooth(image: np.ndarray, epsilon_factor: float) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    binary = (gray > 127).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = np.zeros_like(binary)
    if not contours:
        return out
    largest = max(contours, key=cv2.contourArea)
    epsilon = float(epsilon_factor) * cv2.arcLength(largest, closed=True)
    approx = cv2.approxPolyDP(largest, epsilon, closed=True)
    cv2.drawContours(out, [approx], -1, 255, thickness=-1)
    return out


def _boundary_smooth_code(
    params: dict[str, Any], input_vars: tuple[str, ...], output_var: str
) -> list[str]:
    (image,) = input_vars
    eps = float(params["epsilon_factor"])
    return [
        f"_gray = cv2.cvtColor({image}, cv2.COLOR_BGR2GRAY) if {image}.ndim == 3 else {image}",
        "_binary = (_gray > 127).astype('uint8') * 255",
        "_contours, _ = cv2.findContours(_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)",
        f"{output_var} = np.zeros_like(_binary)",
        "if _contours:",
        "    _largest = max(_contours, key=cv2.contourArea)",
        f"    _epsilon = {eps} * cv2.arcLength(_largest, closed=True)",
        "    _approx = cv2.approxPolyDP(_largest, _epsilon, closed=True)",
        f"    cv2.drawContours({output_var}, [_approx], -1, 255, thickness=-1)",
    ]


BOUNDARY_SMOOTH = OperationSpec(
    id="synthesis.boundary_smooth",
    name="Boundary Smooth",
    category="Synthesis",
    description=(
        "Approximate the largest contour with a polygon (cv2.approxPolyDP) "
        "and re-draw it filled — gives a 'ground-truth defect-free' version "
        "of the input shape. Higher epsilon = stronger smoothing (fewer "
        "vertices, more idealised)."
    ),
    parameters=(
        Parameter(
            name="epsilon_factor", kind="float", default=0.01, min=0.001, max=0.2, step=0.001,
            label="Epsilon factor",
            description=(
                "Fraction of the contour's arc length used as approxPolyDP's "
                "epsilon. Small = follows the original closely; large = "
                "collapses to a few vertices."
            ),
        ),
    ),
    func=_boundary_smooth,
    code_export=_boundary_smooth_code,
)


# ---------------------------------------------------------------- defect_extract


def _to_binary(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return (image > 127).astype(np.uint8) * 255


def _defect_extract(
    defective: np.ndarray, reference: np.ndarray, dilate: int
) -> np.ndarray:
    a = _to_binary(defective)
    b = _to_binary(reference)
    if b.shape != a.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_NEAREST)
    diff = cv2.absdiff(a, b)
    if int(dilate) > 0:
        kernel = np.ones((3, 3), np.uint8)
        diff = cv2.dilate(diff, kernel, iterations=int(dilate))
    return diff


def _defect_extract_code(
    params: dict[str, Any], input_vars: tuple[str, ...], output_var: str
) -> list[str]:
    defective, reference = input_vars
    dilate = int(params["dilate"])
    lines = [
        f"_a = cv2.cvtColor({defective}, cv2.COLOR_BGR2GRAY) if {defective}.ndim == 3 else {defective}",
        "_a = (_a > 127).astype('uint8') * 255",
        f"_b = cv2.cvtColor({reference}, cv2.COLOR_BGR2GRAY) if {reference}.ndim == 3 else {reference}",
        "_b = (_b > 127).astype('uint8') * 255",
        "if _b.shape != _a.shape:",
        "    _b = cv2.resize(_b, (_a.shape[1], _a.shape[0]), interpolation=cv2.INTER_NEAREST)",
        f"{output_var} = cv2.absdiff(_a, _b)",
    ]
    if dilate > 0:
        lines.extend([
            "_kernel = np.ones((3, 3), 'uint8')",
            f"{output_var} = cv2.dilate({output_var}, _kernel, iterations={dilate})",
        ])
    return lines


DEFECT_EXTRACT = OperationSpec(
    id="synthesis.defect_extract",
    name="Defect Extract",
    category="Synthesis",
    description=(
        "Binarised `defective` XOR binarised `reference` = defect-only mask. "
        "Feed the original threshold result into `defective` and a smoothed "
        "version (Boundary Smooth) into `reference` — the difference is the "
        "geometry of the defect alone, ready to feed Seamless Clone or Copy "
        "Paste Defect."
    ),
    parameters=(
        Parameter(
            name="dilate", kind="int", default=0, min=0, max=20, step=1,
            label="Dilate iterations",
            description="Optional dilation to thicken the extracted defect mask.",
        ),
    ),
    func=_defect_extract,
    code_export=_defect_extract_code,
    input_ports=("defective", "reference"),
)


# ------------------------------------------------------------- copy_paste_defect


def _copy_paste_defect(
    background: np.ndarray,
    defect_patch: np.ndarray,
    defect_mask: np.ndarray,
    position_jitter: int,
    rotation_jitter: int,
    scale_jitter: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))

    bg = background.copy()
    mask = defect_mask
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if mask.shape != defect_patch.shape[:2]:
        mask = cv2.resize(
            mask, (defect_patch.shape[1], defect_patch.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    binary = (mask > 127).astype(np.uint8) * 255
    ys, xs = np.where(binary > 0)
    if len(xs) == 0:
        return bg

    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    patch = defect_patch[y0:y1, x0:x1]
    patch_mask = binary[y0:y1, x0:x1]

    angle = float(rng.uniform(-int(rotation_jitter), int(rotation_jitter)))
    scale = float(1.0 + rng.uniform(-float(scale_jitter), float(scale_jitter)))
    scale = max(0.1, scale)

    ph, pw = patch.shape[:2]
    rot = cv2.getRotationMatrix2D((pw / 2, ph / 2), angle, scale)
    cos_a = abs(rot[0, 0])
    sin_a = abs(rot[0, 1])
    new_w = max(1, int(pw * cos_a + ph * sin_a))
    new_h = max(1, int(pw * sin_a + ph * cos_a))
    rot[0, 2] += (new_w - pw) / 2
    rot[1, 2] += (new_h - ph) / 2
    patch_t = cv2.warpAffine(patch, rot, (new_w, new_h), flags=cv2.INTER_LINEAR)
    mask_t = cv2.warpAffine(patch_mask, rot, (new_w, new_h), flags=cv2.INTER_NEAREST)
    mask_t = (mask_t > 127).astype(np.uint8) * 255

    bh, bw = bg.shape[:2]
    jx = int(rng.integers(-int(position_jitter), int(position_jitter) + 1)) if position_jitter > 0 else 0
    jy = int(rng.integers(-int(position_jitter), int(position_jitter) + 1)) if position_jitter > 0 else 0
    cx = bw // 2 + jx
    cy = bh // 2 + jy

    x_tl = cx - new_w // 2
    y_tl = cy - new_h // 2
    bx0 = max(0, x_tl)
    by0 = max(0, y_tl)
    bx1 = min(bw, x_tl + new_w)
    by1 = min(bh, y_tl + new_h)
    if bx1 <= bx0 or by1 <= by0:
        return bg

    px0 = bx0 - x_tl
    py0 = by0 - y_tl
    px1 = px0 + (bx1 - bx0)
    py1 = py0 + (by1 - by0)

    patch_clip = patch_t[py0:py1, px0:px1]
    mask_clip = mask_t[py0:py1, px0:px1]

    # Coerce patch channels to bg's layout so the assignment shape matches.
    if bg.ndim == 3 and patch_clip.ndim == 2:
        patch_clip = cv2.cvtColor(patch_clip, cv2.COLOR_GRAY2BGR)
    elif bg.ndim == 2 and patch_clip.ndim == 3:
        patch_clip = cv2.cvtColor(patch_clip, cv2.COLOR_BGR2GRAY)
    elif bg.ndim == 3 and patch_clip.ndim == 3 and patch_clip.shape[2] != bg.shape[2]:
        if bg.shape[2] == 3 and patch_clip.shape[2] == 4:
            patch_clip = cv2.cvtColor(patch_clip, cv2.COLOR_BGRA2BGR)

    region = bg[by0:by1, bx0:bx1]
    sel = mask_clip > 0
    region[sel] = patch_clip[sel]
    bg[by0:by1, bx0:bx1] = region
    return bg


def _copy_paste_defect_code(
    params: dict[str, Any], input_vars: tuple[str, ...], output_var: str
) -> list[str]:
    bg, patch, mask = input_vars
    pos = int(params["position_jitter"])
    rot = int(params["rotation_jitter"])
    scale = float(params["scale_jitter"])
    seed = int(params["seed"])
    return [
        f"_rng = np.random.default_rng({seed})",
        f"_bg = {bg}.copy()",
        f"_patch_full = {patch}",
        f"_mask = {mask}",
        "if _mask.ndim == 3:",
        "    _mask = cv2.cvtColor(_mask, cv2.COLOR_BGR2GRAY)",
        "if _mask.shape != _patch_full.shape[:2]:",
        "    _mask = cv2.resize(_mask, (_patch_full.shape[1], _patch_full.shape[0]), interpolation=cv2.INTER_NEAREST)",
        "_binary = (_mask > 127).astype('uint8') * 255",
        "_ys, _xs = np.where(_binary > 0)",
        "if len(_xs) == 0:",
        f"    {output_var} = _bg",
        "else:",
        "    _x0, _x1 = int(_xs.min()), int(_xs.max()) + 1",
        "    _y0, _y1 = int(_ys.min()), int(_ys.max()) + 1",
        "    _p = _patch_full[_y0:_y1, _x0:_x1]",
        "    _m = _binary[_y0:_y1, _x0:_x1]",
        f"    _angle = float(_rng.uniform({-rot}, {rot}))",
        f"    _scale = max(0.1, 1.0 + float(_rng.uniform({-scale}, {scale})))",
        "    _ph, _pw = _p.shape[:2]",
        "    _M = cv2.getRotationMatrix2D((_pw / 2, _ph / 2), _angle, _scale)",
        "    _cos = abs(_M[0, 0]); _sin = abs(_M[0, 1])",
        "    _nw = max(1, int(_pw * _cos + _ph * _sin))",
        "    _nh = max(1, int(_pw * _sin + _ph * _cos))",
        "    _M[0, 2] += (_nw - _pw) / 2",
        "    _M[1, 2] += (_nh - _ph) / 2",
        "    _pt = cv2.warpAffine(_p, _M, (_nw, _nh), flags=cv2.INTER_LINEAR)",
        "    _mt = cv2.warpAffine(_m, _M, (_nw, _nh), flags=cv2.INTER_NEAREST)",
        "    _mt = (_mt > 127).astype('uint8') * 255",
        "    _bh, _bw = _bg.shape[:2]",
        f"    _jx = int(_rng.integers({-pos}, {pos + 1})) if {pos} > 0 else 0",
        f"    _jy = int(_rng.integers({-pos}, {pos + 1})) if {pos} > 0 else 0",
        "    _cx = _bw // 2 + _jx",
        "    _cy = _bh // 2 + _jy",
        "    _xtl = _cx - _nw // 2",
        "    _ytl = _cy - _nh // 2",
        "    _bx0 = max(0, _xtl); _by0 = max(0, _ytl)",
        "    _bx1 = min(_bw, _xtl + _nw); _by1 = min(_bh, _ytl + _nh)",
        "    if _bx1 <= _bx0 or _by1 <= _by0:",
        f"        {output_var} = _bg",
        "    else:",
        "        _px0 = _bx0 - _xtl; _py0 = _by0 - _ytl",
        "        _px1 = _px0 + (_bx1 - _bx0); _py1 = _py0 + (_by1 - _by0)",
        "        _pc = _pt[_py0:_py1, _px0:_px1]",
        "        _mc = _mt[_py0:_py1, _px0:_px1]",
        "        if _bg.ndim == 3 and _pc.ndim == 2:",
        "            _pc = cv2.cvtColor(_pc, cv2.COLOR_GRAY2BGR)",
        "        elif _bg.ndim == 2 and _pc.ndim == 3:",
        "            _pc = cv2.cvtColor(_pc, cv2.COLOR_BGR2GRAY)",
        "        _region = _bg[_by0:_by1, _bx0:_bx1]",
        "        _sel = _mc > 0",
        "        _region[_sel] = _pc[_sel]",
        "        _bg[_by0:_by1, _bx0:_bx1] = _region",
        f"        {output_var} = _bg",
    ]


COPY_PASTE_DEFECT = OperationSpec(
    id="synthesis.copy_paste_defect",
    name="Copy Paste Defect",
    category="Synthesis",
    description=(
        "Cut-Paste-Learn style augmentation: crop the masked region of "
        "`defect_patch`, apply rotation / scale / position jitter, and stamp "
        "it onto `background`. The `seed` parameter makes each variant "
        "reproducible — pick a seed to recall a specific augmentation, "
        "vary it (or batch-vary it) to grow a dataset."
    ),
    parameters=(
        Parameter(
            name="position_jitter", kind="int", default=0, min=0, max=500, step=1,
            label="Position jitter (px)",
            description="Max horizontal/vertical offset from background centre.",
        ),
        Parameter(
            name="rotation_jitter", kind="int", default=0, min=0, max=180, step=1,
            label="Rotation jitter (°)",
            description="Max ± rotation in degrees.",
        ),
        Parameter(
            name="scale_jitter", kind="float", default=0.0, min=0.0, max=0.9, step=0.01,
            label="Scale jitter",
            description="Max ± scale multiplier (e.g. 0.2 = 0.8x..1.2x).",
        ),
        Parameter(
            name="seed", kind="int", default=0, min=0, max=999999, step=1,
            label="Seed",
            description="RNG seed — fixes the chosen jitter to make the result reproducible.",
        ),
    ),
    func=_copy_paste_defect,
    code_export=_copy_paste_defect_code,
    input_ports=("background", "defect_patch", "defect_mask"),
)


# ----------------------------------------------------------- perlin_noise_overlay


def _value_noise(
    shape: tuple[int, int], scale: float, octaves: int, rng: np.random.Generator
) -> np.ndarray:
    """Fractal value noise — a cheap Perlin stand-in built from layered
    low-res white noise upsampled with cubic interpolation. Deterministic
    given the rng's seed."""
    h, w = shape
    out = np.zeros((h, w), dtype=np.float32)
    amp = 1.0
    for octave in range(int(octaves)):
        freq = (2 ** octave) / max(1.0, float(scale))
        oh = max(2, int(round(h * freq)))
        ow = max(2, int(round(w * freq)))
        layer = rng.random((oh, ow), dtype=np.float32)
        layer = cv2.resize(layer, (w, h), interpolation=cv2.INTER_CUBIC)
        out += layer * amp
        amp *= 0.5
    return out


def _perlin_noise_overlay(
    image: np.ndarray, scale: float, octaves: int, amplitude: float, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    h, w = image.shape[:2]
    noise = _value_noise((h, w), float(scale), int(octaves), rng)
    noise -= float(noise.mean())
    std = float(noise.std())
    if std > 0:
        noise = noise / std  # type: ignore[assignment]
    delta = noise * float(amplitude) * 127.0
    work = image.astype(np.float32)
    if work.ndim == 3:
        delta = delta[..., None]
    return np.clip(work + delta, 0, 255).astype(np.uint8)


def _perlin_noise_overlay_code(
    params: dict[str, Any], input_vars: tuple[str, ...], output_var: str
) -> list[str]:
    (image,) = input_vars
    scale = float(params["scale"])
    octaves = int(params["octaves"])
    amp = float(params["amplitude"])
    seed = int(params["seed"])
    return [
        f"_rng = np.random.default_rng({seed})",
        f"_h, _w = {image}.shape[:2]",
        "_noise = np.zeros((_h, _w), dtype='float32')",
        "_amp = 1.0",
        f"for _octave in range({octaves}):",
        f"    _freq = (2 ** _octave) / max(1.0, {scale})",
        "    _oh = max(2, int(round(_h * _freq)))",
        "    _ow = max(2, int(round(_w * _freq)))",
        "    _layer = _rng.random((_oh, _ow), dtype='float32')",
        "    _layer = cv2.resize(_layer, (_w, _h), interpolation=cv2.INTER_CUBIC)",
        "    _noise += _layer * _amp",
        "    _amp *= 0.5",
        "_noise -= float(_noise.mean())",
        "_std = float(_noise.std())",
        "if _std > 0:",
        "    _noise = _noise / _std",
        f"_delta = _noise * {amp} * 127.0",
        f"_work = {image}.astype('float32')",
        "if _work.ndim == 3:",
        "    _delta = _delta[..., None]",
        f"{output_var} = np.clip(_work + _delta, 0, 255).astype('uint8')",
    ]


PERLIN_NOISE_OVERLAY = OperationSpec(
    id="synthesis.perlin_noise_overlay",
    name="Perlin Noise Overlay",
    category="Synthesis",
    description=(
        "Add fractal value-noise (a cheap Perlin stand-in) to the input "
        "image. Useful for simulating subtle texture variation — corrosion, "
        "lighting noise, sensor grain — without altering geometry. The "
        "`seed` parameter makes each noise field reproducible."
    ),
    parameters=(
        Parameter(
            name="scale", kind="float", default=8.0, min=1.0, max=100.0, step=0.5,
            label="Scale",
            description="Base feature size — larger = smoother / lower frequency noise.",
        ),
        Parameter(
            name="octaves", kind="int", default=4, min=1, max=8, step=1,
            label="Octaves",
            description="Number of frequency layers stacked. More = richer detail, slower.",
        ),
        Parameter(
            name="amplitude", kind="float", default=0.15, min=0.0, max=1.0, step=0.01,
            label="Amplitude",
            description="Strength of the noise overlay. 0 = no change, 1 = ±127 intensity swing.",
        ),
        Parameter(
            name="seed", kind="int", default=0, min=0, max=999999, step=1,
            label="Seed",
            description="RNG seed — fixes the noise field for reproducible results.",
        ),
    ),
    func=_perlin_noise_overlay,
    code_export=_perlin_noise_overlay_code,
)


# --------------------------------------------------------------- grain_flattening


def _grain_flattening_op(
    image: np.ndarray,
    angle_degrees: float,
    intensity: float,
    feather_fraction: float,
    fill_roi: bool,
    seed: int,
) -> np.ndarray:
    """Pipeline wrapper around ``cvstudio.defects.inject_grain_flattening``.

    Drops the label tuple — the pipeline contract is a single ndarray. Use
    the library function directly when you need the YOLO label metadata.

    When ``fill_roi`` is True the entire input is treated as the defect
    region; combine with cvstudio's mouse-drawn pipeline ROI to flatten
    exactly the area you painted. When False the op picks its own sub-ROI
    centred on a Canny edge pixel — useful for full-image previews."""
    out, _ = inject_grain_flattening(
        image,
        angle_degrees=float(angle_degrees),
        intensity=float(intensity),
        feather_fraction=float(feather_fraction),
        fill_roi=bool(fill_roi),
        seed=int(seed),
    )
    return out


def _grain_flattening_op_code(
    params: dict[str, Any], input_vars: tuple[str, ...], output_var: str
) -> list[str]:
    (image,) = input_vars
    angle = float(params["angle_degrees"])
    intensity = float(params["intensity"])
    feather = float(params["feather_fraction"])
    fill = bool(params["fill_roi"])
    seed = int(params["seed"])
    return [
        "from cvstudio.defects import inject_grain_flattening",
        f"{output_var}, _label = inject_grain_flattening(",
        f"    {image},",
        f"    angle_degrees={angle},",
        f"    intensity={intensity},",
        f"    feather_fraction={feather},",
        f"    fill_roi={fill},",
        f"    seed={seed},",
        ")",
    ]


GRAIN_FLATTENING = OperationSpec(
    id="synthesis.grain_flattening",
    name="Grain Flattening",
    category="Synthesis",
    description=(
        "Inject a synthetic grain-flattening defect — directional motion "
        "blur + morph close + local contrast drop, feathered so there is no "
        "boxy seam. Draw a pipeline ROI on the image (toolbar → Select ROI) "
        "and keep 'Fill ROI' on to flatten exactly that region; turn it off "
        "to let the op auto-pick a small sub-ROI on a Canny edge. Angle is a "
        "free 0°–180° slider, not just horizontal/vertical."
    ),
    parameters=(
        Parameter(
            name="angle_degrees", kind="float", default=0.0, min=0.0, max=180.0, step=1.0,
            label="Angle (°)",
            description="Smear angle. 0° = horizontal, 90° = vertical, anything else = diagonal.",
        ),
        Parameter(
            name="intensity", kind="float", default=0.5, min=0.0, max=1.0, step=0.05,
            label="Intensity",
            description="Severity — drives motion-blur kernel size and contrast drop.",
        ),
        Parameter(
            name="feather_fraction", kind="float", default=0.3, min=0.0, max=0.5, step=0.05,
            label="Feather",
            description="Share of each ROI side that fades into the surround.",
        ),
        Parameter(
            name="fill_roi", kind="bool", default=True,
            label="Fill ROI",
            description=(
                "ON = flatten the entire input (pair with a mouse-drawn "
                "pipeline ROI). OFF = auto-pick a sub-ROI on a Canny edge."
            ),
        ),
        Parameter(
            name="seed", kind="int", default=0, min=0, max=999999, step=1,
            label="Seed",
            description="RNG seed — only consulted when Fill ROI is OFF (auto-ROI sampler).",
        ),
    ),
    func=_grain_flattening_op,
    code_export=_grain_flattening_op_code,
)


ALL: tuple[OperationSpec, ...] = (
    BOUNDARY_EXTRACT,
    BOUNDARY_SMOOTH,
    DEFECT_EXTRACT,
    COPY_PASTE_DEFECT,
    PERLIN_NOISE_OVERLAY,
    GRAIN_FLATTENING,
)
