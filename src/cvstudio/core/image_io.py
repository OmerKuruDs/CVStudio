"""Robust image loader that normalizes higher-bit-depth inputs to uint8.

OpenCV happily decodes 16-bit and floating-point TIFFs / PNGs via
`cv2.imread(..., IMREAD_UNCHANGED)`, but every downstream consumer in the app
(pipeline ops, Qt display, code export) assumes uint8 arrays. A naive
`np.clip(image, 0, 255)` washes such inputs out to near-white. This module
rescales the dynamic range instead so the resulting uint8 image is visually
faithful and feeds into the pipeline without crashes.

The helpers are deliberately Qt-free so they can be reused by both the UI
loader and `core/batch.py`.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_image(path: str | Path) -> np.ndarray | None:
    """Load an image from disk and return it as a uint8 ndarray.

    Returns None when the file cannot be read (missing, unsupported codec,
    corrupted). The result has the same channel count as the file (BGR / BGRA
    / grayscale) — only the dtype is normalized.
    """
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None
    return to_uint8(raw)


def to_uint8(image: np.ndarray) -> np.ndarray:
    """Coerce `image` to uint8, rescaling instead of clipping for higher bit
    depths. Float inputs are min/max normalized; uint16 is right-shifted to
    keep the most significant 8 bits (faster than a divide and matches what a
    histogram preview would show)."""
    if image.dtype == np.uint8:
        return image
    if image.dtype == np.uint16:
        return (image >> 8).astype(np.uint8)
    if image.dtype in (np.float32, np.float64):
        lo = float(np.min(image))
        hi = float(np.max(image))
        if hi > lo:
            scaled = (image.astype(np.float32) - lo) / (hi - lo) * 255.0
            return np.asarray(np.clip(scaled, 0, 255).astype(np.uint8))
        return np.zeros(image.shape, dtype=np.uint8)
    # int16 / int32 / int64 / uint32 — conservative min/max normalisation so
    # the user can at least see *something*.
    as_float = image.astype(np.float32)
    lo_i = float(np.min(as_float))
    hi_i = float(np.max(as_float))
    if hi_i > lo_i:
        scaled = (as_float - lo_i) / (hi_i - lo_i) * 255.0
        return np.asarray(np.clip(scaled, 0, 255).astype(np.uint8))
    return np.zeros(image.shape, dtype=np.uint8)


def read_thumbnail(path: str | Path, *, max_dim: int = 200) -> np.ndarray | None:
    """Fast low-resolution decode for gallery thumbnails. Uses cv2's reduced
    decode flag so very large source files don't pay for a full-resolution
    read just to be downsized. Returns a uint8 ndarray no larger than
    `max_dim` on its longest side, or None on failure."""
    raw = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_2)
    if raw is None:
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None
    img = to_uint8(raw)
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest > max_dim:
        scale = max_dim / longest
        new_w = max(1, round(w * scale))
        new_h = max(1, round(h * scale))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img
