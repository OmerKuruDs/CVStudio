from __future__ import annotations

import numpy as np

from cvstudio.operations.freq import (
    BAND_PASS,
    FFT_MAGNITUDE,
    FFT_PHASE,
    HIGH_PASS,
    LOW_PASS,
)


def _striped_image(size: int = 64, period: int = 8) -> np.ndarray:
    """Vertical stripes — has strong horizontal-frequency content."""
    img = np.zeros((size, size), dtype=np.uint8)
    img[:, ::period] = 255
    return img


def _gradient_image(size: int = 64) -> np.ndarray:
    """Smooth horizontal gradient — almost entirely low-frequency."""
    row = np.linspace(0, 255, size, dtype=np.uint8)
    return np.tile(row, (size, 1))


# ----------------------------------------------------------------- FFT Magnitude


def test_fft_magnitude_returns_uint8_singlechannel() -> None:
    out = FFT_MAGNITUDE.func(_striped_image(), log_scale=True, shift_center=True)
    assert out.dtype == np.uint8
    assert out.ndim == 2
    assert out.shape == (64, 64)


def test_fft_magnitude_accepts_color_input() -> None:
    rgb = np.stack([_striped_image()] * 3, axis=-1)
    out = FFT_MAGNITUDE.func(rgb, log_scale=True, shift_center=True)
    assert out.shape == (64, 64)


def test_fft_magnitude_center_brightest_when_shifted() -> None:
    out = FFT_MAGNITUDE.func(_gradient_image(), log_scale=True, shift_center=True)
    # DC component sits at the centre after fftshift.
    h, w = out.shape
    centre_intensity = int(out[h // 2, w // 2])
    edge_intensity = int(out[0, 0])
    assert centre_intensity > edge_intensity


# --------------------------------------------------------------------- FFT Phase


def test_fft_phase_returns_uint8() -> None:
    out = FFT_PHASE.func(_striped_image(), shift_center=True)
    assert out.dtype == np.uint8
    assert out.ndim == 2


# ------------------------------------------------------------------- Pass filters


def test_low_pass_smooths_pixel_noise() -> None:
    rng = np.random.default_rng(0)
    noisy = rng.integers(0, 256, size=(64, 64), dtype=np.uint8)
    out = LOW_PASS.func(noisy, cutoff_radius=5, filter_type="Gaussian", order=2)
    assert out.dtype == np.uint8
    # Mean absolute neighbour difference is the right "smoothness" measure
    # here — global std is misleading because the result is normalised back
    # to 0-255. Random noise → ~85; a smoothed pass should be dramatically
    # lower.
    before = float(np.abs(np.diff(noisy.astype(np.int16), axis=1)).mean())
    after = float(np.abs(np.diff(out.astype(np.int16), axis=1)).mean())
    assert after < before * 0.5


def test_high_pass_kills_constant_brightness() -> None:
    flat = np.full((64, 64), 128, dtype=np.uint8)
    out = HIGH_PASS.func(flat, cutoff_radius=10, filter_type="Gaussian", order=2)
    # A constant image has only DC; removing it should yield ~0 everywhere.
    assert int(out.max()) <= 5


def test_high_pass_keeps_edges_in_stripes() -> None:
    stripes = _striped_image()
    out = HIGH_PASS.func(stripes, cutoff_radius=5, filter_type="Ideal", order=2)
    # The stripes' high-frequency content should survive.
    assert int(out.max()) > 50


def test_band_pass_clamps_inner_below_outer() -> None:
    # If user fat-fingers inner > outer, op should not crash.
    stripes = _striped_image()
    out = BAND_PASS.func(
        stripes, inner_radius=20, outer_radius=10, filter_type="Gaussian", order=2
    )
    assert out.shape == stripes.shape
    assert out.dtype == np.uint8


def test_butterworth_order_changes_result() -> None:
    stripes = _striped_image()
    soft = LOW_PASS.func(stripes, cutoff_radius=8, filter_type="Butterworth", order=1)
    sharp = LOW_PASS.func(stripes, cutoff_radius=8, filter_type="Butterworth", order=8)
    # Different orders should produce different filtered images.
    assert not np.array_equal(soft, sharp)
