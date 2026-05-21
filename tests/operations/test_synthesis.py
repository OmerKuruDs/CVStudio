from __future__ import annotations

import cv2
import numpy as np

from cvstudio.operations.synthesis import (
    BOUNDARY_EXTRACT,
    BOUNDARY_SMOOTH,
    COPY_PASTE_DEFECT,
    DEFECT_EXTRACT,
    PERLIN_NOISE_OVERLAY,
)


# ----------------------------------------------------------------- spec sanity


def test_specs_register_under_synthesis_category() -> None:
    for spec in (
        BOUNDARY_EXTRACT,
        BOUNDARY_SMOOTH,
        DEFECT_EXTRACT,
        COPY_PASTE_DEFECT,
        PERLIN_NOISE_OVERLAY,
    ):
        assert spec.category == "Synthesis"
        assert spec.id.startswith("synthesis.")


def test_multi_input_ports_declared() -> None:
    assert DEFECT_EXTRACT.input_ports == ("defective", "reference")
    assert COPY_PASTE_DEFECT.input_ports == ("background", "defect_patch", "defect_mask")


# --------------------------------------------------------------- boundary ops


def _square_image() -> np.ndarray:
    """64x64 image with a white square in the middle on black."""
    img = np.zeros((64, 64), dtype=np.uint8)
    img[20:44, 20:44] = 255
    return img


def test_boundary_extract_draws_only_the_outline() -> None:
    out = BOUNDARY_EXTRACT.func(_square_image(), thickness=1)
    # Centre of the original square is now black again (only outline kept).
    assert int(out[32, 32]) == 0
    # The square's edge pixels are non-zero.
    assert int(out[20, 32]) == 255


def test_boundary_extract_thickness_increases_line_pixels() -> None:
    thin = BOUNDARY_EXTRACT.func(_square_image(), thickness=1)
    thick = BOUNDARY_EXTRACT.func(_square_image(), thickness=5)
    assert int((thick > 0).sum()) > int((thin > 0).sum())


def test_boundary_smooth_returns_a_filled_polygon_approx() -> None:
    out = BOUNDARY_SMOOTH.func(_square_image(), epsilon_factor=0.05)
    # The result is still filled (centre is white) but with simplified
    # boundary geometry; for a perfect square the approximation matches.
    assert int(out[32, 32]) == 255


def test_boundary_smooth_on_empty_input_returns_empty() -> None:
    empty = np.zeros((32, 32), dtype=np.uint8)
    out = BOUNDARY_SMOOTH.func(empty, epsilon_factor=0.01)
    assert int(out.sum()) == 0


# --------------------------------------------------------------- defect_extract


def test_defect_extract_returns_xor_of_binarised_inputs() -> None:
    a = np.zeros((20, 20), dtype=np.uint8)
    a[5:15, 5:15] = 255          # solid square
    b = a.copy()
    b[7:9, 7:9] = 0              # b has a small hole that a does not
    out = DEFECT_EXTRACT.func(a, b, dilate=0)
    # The difference region matches the hole.
    assert int(out[8, 8]) == 255
    # Outside the hole / inside the unchanged square = 0.
    assert int(out[10, 10]) == 0
    assert int(out[0, 0]) == 0


def test_defect_extract_dilate_grows_the_difference() -> None:
    a = np.zeros((20, 20), dtype=np.uint8)
    a[10, 10] = 255
    b = np.zeros((20, 20), dtype=np.uint8)
    raw = DEFECT_EXTRACT.func(a, b, dilate=0)
    dilated = DEFECT_EXTRACT.func(a, b, dilate=2)
    assert int((dilated > 0).sum()) > int((raw > 0).sum())


def test_defect_extract_resizes_mismatched_reference() -> None:
    a = np.zeros((20, 20), dtype=np.uint8)
    a[5:15, 5:15] = 255
    b = np.zeros((10, 10), dtype=np.uint8)
    out = DEFECT_EXTRACT.func(a, b, dilate=0)
    assert out.shape == a.shape


# ----------------------------------------------------------- copy_paste_defect


def _patch_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bg = np.full((64, 64, 3), 80, dtype=np.uint8)
    patch = np.zeros((64, 64, 3), dtype=np.uint8)
    patch[20:40, 20:40] = (200, 100, 50)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[20:40, 20:40] = 255
    return bg, patch, mask


def test_copy_paste_defect_stamps_patch_onto_background() -> None:
    bg, patch, mask = _patch_inputs()
    out = COPY_PASTE_DEFECT.func(
        bg, patch, mask,
        position_jitter=0, rotation_jitter=0, scale_jitter=0.0, seed=0,
    )
    # Background centre took the patch's colour; corners are still the bg.
    assert tuple(int(c) for c in out[32, 32]) == (200, 100, 50)
    assert tuple(int(c) for c in out[0, 0]) == (80, 80, 80)


def test_copy_paste_defect_is_deterministic_for_same_seed() -> None:
    bg, patch, mask = _patch_inputs()
    out_a = COPY_PASTE_DEFECT.func(
        bg, patch, mask,
        position_jitter=10, rotation_jitter=30, scale_jitter=0.2, seed=42,
    )
    out_b = COPY_PASTE_DEFECT.func(
        bg, patch, mask,
        position_jitter=10, rotation_jitter=30, scale_jitter=0.2, seed=42,
    )
    assert np.array_equal(out_a, out_b)


def test_copy_paste_defect_different_seeds_produce_different_results() -> None:
    bg, patch, mask = _patch_inputs()
    out_a = COPY_PASTE_DEFECT.func(
        bg, patch, mask,
        position_jitter=10, rotation_jitter=30, scale_jitter=0.2, seed=1,
    )
    out_b = COPY_PASTE_DEFECT.func(
        bg, patch, mask,
        position_jitter=10, rotation_jitter=30, scale_jitter=0.2, seed=2,
    )
    assert not np.array_equal(out_a, out_b)


def test_copy_paste_defect_with_empty_mask_returns_background_unchanged() -> None:
    bg, patch, _ = _patch_inputs()
    empty = np.zeros((64, 64), dtype=np.uint8)
    out = COPY_PASTE_DEFECT.func(
        bg, patch, empty,
        position_jitter=0, rotation_jitter=0, scale_jitter=0.0, seed=0,
    )
    assert np.array_equal(out, bg)


# --------------------------------------------------------- perlin_noise_overlay


def test_perlin_noise_overlay_changes_pixels_but_keeps_shape() -> None:
    img = np.full((48, 48, 3), 128, dtype=np.uint8)
    out = PERLIN_NOISE_OVERLAY.func(img, scale=8.0, octaves=4, amplitude=0.3, seed=0)
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    # A uniform input + non-zero amplitude must produce variation.
    assert out.std() > 0


def test_perlin_noise_overlay_zero_amplitude_is_a_no_op() -> None:
    img = np.full((32, 32), 128, dtype=np.uint8)
    out = PERLIN_NOISE_OVERLAY.func(img, scale=8.0, octaves=4, amplitude=0.0, seed=0)
    assert np.array_equal(out, img)


def test_perlin_noise_overlay_is_deterministic_for_same_seed() -> None:
    img = np.full((32, 32), 128, dtype=np.uint8)
    a = PERLIN_NOISE_OVERLAY.func(img, scale=8.0, octaves=4, amplitude=0.3, seed=7)
    b = PERLIN_NOISE_OVERLAY.func(img, scale=8.0, octaves=4, amplitude=0.3, seed=7)
    assert np.array_equal(a, b)


def test_perlin_noise_overlay_different_seeds_produce_different_noise() -> None:
    img = np.full((32, 32), 128, dtype=np.uint8)
    a = PERLIN_NOISE_OVERLAY.func(img, scale=8.0, octaves=4, amplitude=0.3, seed=1)
    b = PERLIN_NOISE_OVERLAY.func(img, scale=8.0, octaves=4, amplitude=0.3, seed=2)
    assert not np.array_equal(a, b)


# ------------------------------------------------------ code-export round-trip


def _run_code(spec, params: dict, inputs: dict, output_var: str = "out") -> np.ndarray:
    assert spec.code_export is not None
    lines = spec.code_export(params, tuple(inputs.keys()), output_var)
    namespace = {**inputs, "cv2": cv2, "np": np}
    exec("\n".join(lines), namespace)
    return namespace[output_var]


def test_boundary_extract_code_export_round_trip() -> None:
    img = _square_image()
    out = _run_code(BOUNDARY_EXTRACT, {"thickness": 2}, {"img": img})
    assert out.shape == img.shape


def test_boundary_smooth_code_export_round_trip() -> None:
    img = _square_image()
    out = _run_code(BOUNDARY_SMOOTH, {"epsilon_factor": 0.05}, {"img": img})
    assert out.shape == img.shape


def test_defect_extract_code_export_round_trip() -> None:
    a = np.zeros((20, 20), dtype=np.uint8)
    a[5:15, 5:15] = 255
    b = a.copy()
    b[7:9, 7:9] = 0
    out = _run_code(DEFECT_EXTRACT, {"dilate": 1}, {"a": a, "b": b})
    assert out.shape == a.shape


def test_copy_paste_defect_code_export_round_trip() -> None:
    bg, patch, mask = _patch_inputs()
    out = _run_code(
        COPY_PASTE_DEFECT,
        {"position_jitter": 5, "rotation_jitter": 15, "scale_jitter": 0.1, "seed": 42},
        {"bg": bg, "patch": patch, "mask": mask},
    )
    assert out.shape == bg.shape


def test_perlin_noise_overlay_code_export_round_trip() -> None:
    img = np.full((32, 32, 3), 128, dtype=np.uint8)
    out = _run_code(
        PERLIN_NOISE_OVERLAY,
        {"scale": 8.0, "octaves": 3, "amplitude": 0.2, "seed": 5},
        {"img": img},
    )
    assert out.shape == img.shape
