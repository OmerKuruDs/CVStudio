"""Frequency-domain operations: FFT spectrum, low/high/band-pass filters.

Every op coerces input to grayscale float32 internally and returns a single-
channel uint8 image (the spectrum, or the filtered image converted back from
frequency space). For colour inputs we just process the luminance — the
inverse-FFT outputs are naturally single-channel.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from cvsandbox.core.operation import OperationSpec, Parameter

_FILTER_TYPES = ("Ideal", "Gaussian", "Butterworth")


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def _to_gray_line(in_var: str, out_var: str) -> str:
    return (
        f"{out_var} = cv2.cvtColor({in_var}, cv2.COLOR_BGR2GRAY) "
        f"if {in_var}.ndim == 3 else {in_var}"
    )


def _normalize_uint8(arr: np.ndarray) -> np.ndarray:
    return cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


# ----------------------------------------------------------------- FFT Magnitude


def _fft_magnitude(image: np.ndarray, log_scale: bool, shift_center: bool) -> np.ndarray:
    gray = _to_gray(image).astype(np.float32)
    spectrum = np.fft.fft2(gray)
    if shift_center:
        spectrum = np.fft.fftshift(spectrum)
    mag = np.abs(spectrum)
    if log_scale:
        mag = np.log1p(mag)
    return _normalize_uint8(mag)


def _fft_magnitude_code(
    params: dict[str, Any], input_vars: tuple[str, ...], output_var: str
) -> list[str]:
    (a,) = input_vars
    log = bool(params["log_scale"])
    shift = bool(params["shift_center"])
    lines = [
        _to_gray_line(a, "_gray"),
        "_spectrum = np.fft.fft2(_gray.astype(np.float32))",
    ]
    if shift:
        lines.append("_spectrum = np.fft.fftshift(_spectrum)")
    lines.append("_mag = np.abs(_spectrum)")
    if log:
        lines.append("_mag = np.log1p(_mag)")
    lines.append(f"{output_var} = cv2.normalize(_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)")
    return lines


FFT_MAGNITUDE = OperationSpec(
    id="freq.fft_magnitude",
    name="FFT Magnitude",
    category="Frequency",
    description=(
        "Discrete Fourier transform magnitude as a single-channel image. "
        "Toggle <i>log scale</i> for the conventional astronomy-style spectrum "
        "and <i>shift center</i> to bring DC to the middle."
    ),
    parameters=(
        Parameter(name="log_scale", kind="bool", default=True, label="Log scale"),
        Parameter(name="shift_center", kind="bool", default=True, label="Shift center"),
    ),
    func=_fft_magnitude,
    code_export=_fft_magnitude_code,
)


# --------------------------------------------------------------------- FFT Phase


def _fft_phase(image: np.ndarray, shift_center: bool) -> np.ndarray:
    gray = _to_gray(image).astype(np.float32)
    spectrum = np.fft.fft2(gray)
    if shift_center:
        spectrum = np.fft.fftshift(spectrum)
    phase = np.angle(spectrum)  # range −π .. π
    return _normalize_uint8(phase)


def _fft_phase_code(
    params: dict[str, Any], input_vars: tuple[str, ...], output_var: str
) -> list[str]:
    (a,) = input_vars
    shift = bool(params["shift_center"])
    lines = [
        _to_gray_line(a, "_gray"),
        "_spectrum = np.fft.fft2(_gray.astype(np.float32))",
    ]
    if shift:
        lines.append("_spectrum = np.fft.fftshift(_spectrum)")
    lines.append("_phase = np.angle(_spectrum)")
    lines.append(f"{output_var} = cv2.normalize(_phase, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)")
    return lines


FFT_PHASE = OperationSpec(
    id="freq.fft_phase",
    name="FFT Phase",
    category="Frequency",
    description=(
        "Phase component of the discrete Fourier transform, normalised to "
        "0-255. Often visually noisy but carries most of the structural info."
    ),
    parameters=(
        Parameter(name="shift_center", kind="bool", default=True, label="Shift center"),
    ),
    func=_fft_phase,
    code_export=_fft_phase_code,
)


# ------------------------------------------------------ Frequency-domain filters


def _circular_mask(shape: tuple[int, int], radius: float, filter_type: str, order: int) -> np.ndarray:
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    dist = np.sqrt((x - ccol) ** 2 + (y - crow) ** 2).astype(np.float32)
    r = max(1.0, float(radius))
    if filter_type == "Ideal":
        return (dist <= r).astype(np.float32)
    if filter_type == "Gaussian":
        return np.exp(-(dist**2) / (2 * (r**2))).astype(np.float32)
    # Butterworth
    return (1.0 / (1.0 + (dist / r) ** (2 * max(1, int(order))))).astype(np.float32)


def _apply_mask_in_frequency(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    f = np.fft.fft2(gray.astype(np.float32))
    f_shifted = np.fft.fftshift(f)
    filtered = f_shifted * mask
    back = np.fft.ifftshift(filtered)
    inv = np.abs(np.fft.ifft2(back))
    return _normalize_uint8(inv)


def _low_pass(
    image: np.ndarray, cutoff_radius: int, filter_type: str, order: int
) -> np.ndarray:
    gray = _to_gray(image)
    mask = _circular_mask(gray.shape[:2], float(cutoff_radius), filter_type, int(order))
    return _apply_mask_in_frequency(gray, mask)


def _high_pass(
    image: np.ndarray, cutoff_radius: int, filter_type: str, order: int
) -> np.ndarray:
    gray = _to_gray(image)
    low_mask = _circular_mask(gray.shape[:2], float(cutoff_radius), filter_type, int(order))
    return _apply_mask_in_frequency(gray, 1.0 - low_mask)


def _band_pass(
    image: np.ndarray,
    inner_radius: int,
    outer_radius: int,
    filter_type: str,
    order: int,
) -> np.ndarray:
    gray = _to_gray(image)
    inner = max(0, int(inner_radius))
    outer = max(inner + 1, int(outer_radius))
    low_outer = _circular_mask(gray.shape[:2], float(outer), filter_type, int(order))
    low_inner = _circular_mask(gray.shape[:2], float(inner), filter_type, int(order))
    return _apply_mask_in_frequency(gray, low_outer - low_inner)


def _emit_pass_filter(
    a: str,
    output_var: str,
    radius: int,
    filter_type: str,
    order: int,
    invert: bool,
) -> list[str]:
    return [
        _to_gray_line(a, "_gray"),
        "_rows, _cols = _gray.shape[:2]",
        "_crow, _ccol = _rows // 2, _cols // 2",
        "_y, _x = np.ogrid[:_rows, :_cols]",
        "_dist = np.sqrt((_x - _ccol) ** 2 + (_y - _crow) ** 2).astype(np.float32)",
        f"_r = max(1.0, {float(radius)})",
        f"_ft = {filter_type!r}",
        "if _ft == 'Ideal':",
        "    _mask = (_dist <= _r).astype(np.float32)",
        "elif _ft == 'Gaussian':",
        "    _mask = np.exp(-(_dist ** 2) / (2 * (_r ** 2))).astype(np.float32)",
        "else:",
        f"    _mask = (1.0 / (1.0 + (_dist / _r) ** (2 * max(1, {int(order)})))).astype(np.float32)",
        ("_mask = 1.0 - _mask" if invert else "# (low-pass: keep mask as is)"),
        "_f = np.fft.fftshift(np.fft.fft2(_gray.astype(np.float32)))",
        "_inv = np.abs(np.fft.ifft2(np.fft.ifftshift(_f * _mask)))",
        f"{output_var} = cv2.normalize(_inv, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)",
    ]


def _low_pass_code(
    params: dict[str, Any], input_vars: tuple[str, ...], output_var: str
) -> list[str]:
    (a,) = input_vars
    return _emit_pass_filter(
        a,
        output_var,
        int(params["cutoff_radius"]),
        params["filter_type"],
        int(params["order"]),
        invert=False,
    )


def _high_pass_code(
    params: dict[str, Any], input_vars: tuple[str, ...], output_var: str
) -> list[str]:
    (a,) = input_vars
    return _emit_pass_filter(
        a,
        output_var,
        int(params["cutoff_radius"]),
        params["filter_type"],
        int(params["order"]),
        invert=True,
    )


def _band_pass_code(
    params: dict[str, Any], input_vars: tuple[str, ...], output_var: str
) -> list[str]:
    (a,) = input_vars
    inner = max(0, int(params["inner_radius"]))
    outer = max(inner + 1, int(params["outer_radius"]))
    ft = params["filter_type"]
    order = int(params["order"])
    return [
        _to_gray_line(a, "_gray"),
        "_rows, _cols = _gray.shape[:2]",
        "_crow, _ccol = _rows // 2, _cols // 2",
        "_y, _x = np.ogrid[:_rows, :_cols]",
        "_dist = np.sqrt((_x - _ccol) ** 2 + (_y - _crow) ** 2).astype(np.float32)",
        f"_ri = max(1.0, {float(inner)})",
        f"_ro = max(1.0, {float(outer)})",
        f"_ft = {ft!r}",
        "if _ft == 'Ideal':",
        "    _inner = (_dist <= _ri).astype(np.float32)",
        "    _outer = (_dist <= _ro).astype(np.float32)",
        "elif _ft == 'Gaussian':",
        "    _inner = np.exp(-(_dist ** 2) / (2 * (_ri ** 2))).astype(np.float32)",
        "    _outer = np.exp(-(_dist ** 2) / (2 * (_ro ** 2))).astype(np.float32)",
        "else:",
        f"    _o = max(1, {order})",
        "    _inner = (1.0 / (1.0 + (_dist / _ri) ** (2 * _o))).astype(np.float32)",
        "    _outer = (1.0 / (1.0 + (_dist / _ro) ** (2 * _o))).astype(np.float32)",
        "_mask = _outer - _inner",
        "_f = np.fft.fftshift(np.fft.fft2(_gray.astype(np.float32)))",
        "_inv = np.abs(np.fft.ifft2(np.fft.ifftshift(_f * _mask)))",
        f"{output_var} = cv2.normalize(_inv, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)",
    ]


LOW_PASS = OperationSpec(
    id="freq.low_pass",
    name="Low-Pass Filter",
    category="Frequency",
    description=(
        "Frequency-domain low-pass: zeros out spectrum components further than "
        "<i>cutoff radius</i> from the centre. Use Gaussian or Butterworth to "
        "avoid the ringing artefacts of an ideal cutoff."
    ),
    parameters=(
        Parameter(
            name="cutoff_radius",
            kind="int",
            default=30,
            min=1,
            max=500,
            label="Cutoff radius (px)",
        ),
        Parameter(
            name="filter_type",
            kind="choice",
            default="Gaussian",
            choices=_FILTER_TYPES,
            label="Filter shape",
        ),
        Parameter(
            name="order",
            kind="int",
            default=2,
            min=1,
            max=8,
            label="Butterworth order",
            description="Only used when filter shape = Butterworth.",
        ),
    ),
    func=_low_pass,
    code_export=_low_pass_code,
)


HIGH_PASS = OperationSpec(
    id="freq.high_pass",
    name="High-Pass Filter",
    category="Frequency",
    description=(
        "Frequency-domain high-pass: keeps spectrum components further than "
        "<i>cutoff radius</i> from the centre. Highlights fine detail and "
        "edges while removing slow brightness gradients."
    ),
    parameters=(
        Parameter(
            name="cutoff_radius",
            kind="int",
            default=30,
            min=1,
            max=500,
            label="Cutoff radius (px)",
        ),
        Parameter(
            name="filter_type",
            kind="choice",
            default="Gaussian",
            choices=_FILTER_TYPES,
            label="Filter shape",
        ),
        Parameter(
            name="order",
            kind="int",
            default=2,
            min=1,
            max=8,
            label="Butterworth order",
            description="Only used when filter shape = Butterworth.",
        ),
    ),
    func=_high_pass,
    code_export=_high_pass_code,
)


BAND_PASS = OperationSpec(
    id="freq.band_pass",
    name="Band-Pass Filter",
    category="Frequency",
    description=(
        "Keeps only the annular ring of frequencies between <i>inner</i> and "
        "<i>outer</i> radii. Useful for isolating a specific scale of detail."
    ),
    parameters=(
        Parameter(
            name="inner_radius",
            kind="int",
            default=10,
            min=0,
            max=500,
            label="Inner radius (px)",
        ),
        Parameter(
            name="outer_radius",
            kind="int",
            default=60,
            min=1,
            max=500,
            label="Outer radius (px)",
        ),
        Parameter(
            name="filter_type",
            kind="choice",
            default="Gaussian",
            choices=_FILTER_TYPES,
            label="Filter shape",
        ),
        Parameter(
            name="order",
            kind="int",
            default=2,
            min=1,
            max=8,
            label="Butterworth order",
        ),
    ),
    func=_band_pass,
    code_export=_band_pass_code,
)


ALL: tuple[OperationSpec, ...] = (
    FFT_MAGNITUDE,
    FFT_PHASE,
    LOW_PASS,
    HIGH_PASS,
    BAND_PASS,
)
