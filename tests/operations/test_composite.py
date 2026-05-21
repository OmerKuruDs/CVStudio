from __future__ import annotations

import numpy as np

from cvstudio.operations.composite import (
    APPLY_MASK,
    BLEND,
    DIFFERENCE,
    INPAINT,
    MASK_PASTE,
    SEAMLESS_CLONE,
)


def test_blend_default_alpha_is_a_midpoint_average() -> None:
    a = np.full((4, 4, 3), 80, dtype=np.uint8)
    b = np.full((4, 4, 3), 200, dtype=np.uint8)
    out = BLEND.func(a, b, alpha=0.5)
    assert int(out[0, 0, 0]) == 140  # (80 + 200) / 2


def test_blend_alpha_zero_returns_a_only() -> None:
    a = np.full((4, 4, 3), 50, dtype=np.uint8)
    b = np.full((4, 4, 3), 200, dtype=np.uint8)
    out = BLEND.func(a, b, alpha=0.0)
    assert int(out[0, 0, 0]) == 50


def test_blend_promotes_grayscale_b_to_match_a() -> None:
    a = np.full((4, 4, 3), 100, dtype=np.uint8)
    b_gray = np.full((4, 4), 200, dtype=np.uint8)
    out = BLEND.func(a, b_gray, alpha=0.5)
    assert out.shape == a.shape
    assert int(out[0, 0, 0]) == 150


def test_blend_resizes_b_to_match_a() -> None:
    a = np.full((6, 6, 3), 100, dtype=np.uint8)
    b = np.full((3, 3, 3), 200, dtype=np.uint8)
    out = BLEND.func(a, b, alpha=0.5)
    assert out.shape == a.shape


def test_apply_mask_zeros_pixels_where_mask_is_zero() -> None:
    img = np.full((4, 4, 3), 200, dtype=np.uint8)
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 255
    out = APPLY_MASK.func(img, mask)
    assert int(out[0, 0, 0]) == 0
    assert int(out[2, 2, 0]) == 200


def test_apply_mask_accepts_color_mask() -> None:
    img = np.full((4, 4, 3), 50, dtype=np.uint8)
    color_mask = np.zeros((4, 4, 3), dtype=np.uint8)
    color_mask[..., 0] = 100  # blue channel non-zero
    out = APPLY_MASK.func(img, color_mask)
    # cvtColor(BGR2GRAY) keeps non-zero pixels — every pixel becomes a keep pixel.
    assert (out == img).all()


def test_difference_returns_abs_pixel_delta() -> None:
    a = np.full((4, 4, 3), 100, dtype=np.uint8)
    b = np.full((4, 4, 3), 130, dtype=np.uint8)
    out = DIFFERENCE.func(a, b)
    assert int(out[0, 0, 0]) == 30


def test_difference_is_symmetric() -> None:
    a = np.full((4, 4, 3), 100, dtype=np.uint8)
    b = np.full((4, 4, 3), 200, dtype=np.uint8)
    assert int(DIFFERENCE.func(a, b)[0, 0, 0]) == int(DIFFERENCE.func(b, a)[0, 0, 0])


def test_composite_specs_declare_two_input_ports() -> None:
    assert BLEND.input_ports == ("a", "b")
    assert APPLY_MASK.input_ports == ("image", "mask")
    assert DIFFERENCE.input_ports == ("a", "b")
    assert INPAINT.input_ports == ("image", "mask")
    assert MASK_PASTE.input_ports == ("dst", "mask")
    assert SEAMLESS_CLONE.input_ports == ("src", "dst", "mask")


# --------------------------------------------------------------------- inpaint


def test_inpaint_reconstructs_masked_pixels_from_neighbours() -> None:
    img = np.full((20, 20, 3), 200, dtype=np.uint8)
    # Punch a black hole and tell inpaint to repair it.
    img[8:12, 8:12] = 0
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[8:12, 8:12] = 255
    out = INPAINT.func(img, mask, method="telea", radius=3)
    # The hole should be filled with values close to the surrounding 200 — the
    # surviving border guarantees the neighbourhood is uniform, so reconstructed
    # pixels must land far from black.
    assert int(out[10, 10, 0]) > 150


def test_inpaint_ns_and_telea_both_run() -> None:
    img = np.full((16, 16, 3), 120, dtype=np.uint8)
    img[6:10, 6:10] = 0
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[6:10, 6:10] = 255
    out_telea = INPAINT.func(img, mask, method="telea", radius=2)
    out_ns = INPAINT.func(img, mask, method="ns", radius=2)
    assert out_telea.shape == img.shape
    assert out_ns.shape == img.shape
    assert int(out_telea[8, 8, 0]) > 50
    assert int(out_ns[8, 8, 0]) > 50


def test_inpaint_accepts_color_mask_and_resizes_to_image() -> None:
    img = np.full((12, 12, 3), 180, dtype=np.uint8)
    img[5:7, 5:7] = 0
    # 3-channel mask at a different resolution — must be normalised before use.
    color_mask = np.zeros((6, 6, 3), dtype=np.uint8)
    color_mask[2:4, 2:4, 0] = 255
    out = INPAINT.func(img, color_mask, method="telea", radius=2)
    assert out.shape == img.shape


def test_inpaint_coerces_bgra_input_to_bgr() -> None:
    bgra = np.full((10, 10, 4), 200, dtype=np.uint8)
    bgra[4:6, 4:6, :3] = 0
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[4:6, 4:6] = 255
    out = INPAINT.func(bgra, mask, method="telea", radius=2)
    # Result drops the alpha channel — cv2.inpaint cannot consume BGRA.
    assert out.ndim == 3 and out.shape[2] == 3


# ------------------------------------------------------------------ mask_paste


def test_mask_paste_stamps_fill_value_where_mask_is_set() -> None:
    dst = np.full((4, 4, 3), 200, dtype=np.uint8)
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 255
    out = MASK_PASTE.func(dst, mask, binarize_threshold=127, fill_value=0)
    assert int(out[0, 0, 0]) == 200  # untouched
    assert int(out[1, 1, 0]) == 0    # painted
    assert int(out[2, 2, 0]) == 0


def test_mask_paste_leaves_dst_below_threshold_alone() -> None:
    dst = np.full((4, 4, 3), 200, dtype=np.uint8)
    # Mask values at 100 — below the default 127 threshold, so nothing paints.
    mask = np.full((4, 4), 100, dtype=np.uint8)
    out = MASK_PASTE.func(dst, mask, binarize_threshold=127, fill_value=0)
    assert (out == dst).all()


def test_mask_paste_works_on_grayscale_dst() -> None:
    dst = np.full((4, 4), 150, dtype=np.uint8)
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[2, 2] = 255
    out = MASK_PASTE.func(dst, mask, binarize_threshold=127, fill_value=20)
    assert int(out[2, 2]) == 20
    assert int(out[0, 0]) == 150


def test_mask_paste_resizes_mask_to_dst() -> None:
    dst = np.full((8, 8, 3), 200, dtype=np.uint8)
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 255
    out = MASK_PASTE.func(dst, mask, binarize_threshold=127, fill_value=0)
    assert out.shape == dst.shape
    # Painted region survives the nearest-neighbour upscale.
    assert (out == 0).any()


# ------------------------------------------------------- code-export smoke


def test_inpaint_code_export_round_trips_through_exec() -> None:
    import cv2

    img = np.full((10, 10, 3), 200, dtype=np.uint8)
    img[4:6, 4:6] = 0
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[4:6, 4:6] = 255
    assert INPAINT.code_export is not None
    lines = INPAINT.code_export(
        {"method": "telea", "radius": 2}, ("img", "mask"), "out"
    )
    namespace = {"img": img, "mask": mask, "cv2": cv2, "np": np}
    exec("\n".join(lines), namespace)
    assert namespace["out"].shape == img.shape


def test_mask_paste_code_export_round_trips_through_exec() -> None:
    import cv2

    dst = np.full((4, 4, 3), 200, dtype=np.uint8)
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 255
    assert MASK_PASTE.code_export is not None
    lines = MASK_PASTE.code_export(
        {"binarize_threshold": 127, "fill_value": 0}, ("dst", "mask"), "out"
    )
    namespace = {"dst": dst, "mask": mask, "cv2": cv2, "np": np}
    exec("\n".join(lines), namespace)
    assert int(namespace["out"][1, 1, 0]) == 0
    assert int(namespace["out"][0, 0, 0]) == 200


# --------------------------------------------------------------- seamless_clone


def _seamless_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """src has a sharp gradient inside the mask (so seamlessClone has
    something to transfer — NORMAL_CLONE preserves gradients, not colours,
    so a flat patch would produce no visible change). dst is uniform grey.
    Both 64x64 so the centred patch barely fits after clamp."""
    src = np.full((64, 64, 3), 80, dtype=np.uint8)
    # Hard step inside the mask: top half white, bottom half black — gives a
    # strong horizontal gradient the Poisson solver must propagate into dst.
    src[24:32, 24:40] = (240, 240, 240)
    src[32:40, 24:40] = (20, 20, 20)
    dst = np.full((64, 64, 3), 120, dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[24:40, 24:40] = 255
    return src, dst, mask


def test_seamless_clone_changes_dst_inside_mask_region() -> None:
    src, dst, mask = _seamless_inputs()
    out = SEAMLESS_CLONE.func(
        src, dst, mask,
        mode="normal", auto_center=True, center_x=0, center_y=0,
    )
    assert out.shape == dst.shape
    assert out.dtype == np.uint8
    # The patch carries a strong top-to-bottom gradient — after Poisson blend
    # the upper masked rows must end up brighter than the lower ones, even
    # though dst was uniform grey.
    upper_row = int(out[26, 32, 0])
    lower_row = int(out[38, 32, 0])
    assert upper_row > lower_row + 30, (
        f"expected gradient transfer (upper > lower + 30), got "
        f"upper={upper_row} lower={lower_row}"
    )


def test_seamless_clone_with_empty_mask_returns_dst_unchanged() -> None:
    src, dst, _ = _seamless_inputs()
    empty_mask = np.zeros((64, 64), dtype=np.uint8)
    out = SEAMLESS_CLONE.func(
        src, dst, empty_mask,
        mode="normal", auto_center=True, center_x=0, center_y=0,
    )
    # An empty mask gives the centroid path nothing to anchor on; we return
    # dst (BGR-coerced) unchanged so the pipeline keeps producing a valid image.
    assert np.array_equal(out, dst)


def test_seamless_clone_clamps_out_of_bounds_centre() -> None:
    """Manual centre way outside dst would crash cv2.seamlessClone; the op
    must clamp it so the patch always fits."""
    src, dst, mask = _seamless_inputs()
    out = SEAMLESS_CLONE.func(
        src, dst, mask,
        mode="normal", auto_center=False, center_x=99999, center_y=-50,
    )
    assert out.shape == dst.shape
    assert out.dtype == np.uint8


def test_seamless_clone_promotes_grayscale_inputs_to_bgr() -> None:
    src_gray = np.full((64, 64), 80, dtype=np.uint8)
    src_gray[24:40, 24:40] = 220
    dst_gray = np.full((64, 64), 120, dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[24:40, 24:40] = 255
    out = SEAMLESS_CLONE.func(
        src_gray, dst_gray, mask,
        mode="normal", auto_center=True, center_x=0, center_y=0,
    )
    # Output is BGR even though inputs were gray.
    assert out.ndim == 3 and out.shape[2] == 3


def test_seamless_clone_modes_all_run() -> None:
    src, dst, mask = _seamless_inputs()
    for mode in ("normal", "mixed", "monochrome_transfer"):
        out = SEAMLESS_CLONE.func(
            src, dst, mask,
            mode=mode, auto_center=True, center_x=0, center_y=0,
        )
        assert out.shape == dst.shape, f"{mode} produced wrong shape"


def test_seamless_clone_code_export_round_trips_through_exec() -> None:
    import cv2

    src, dst, mask = _seamless_inputs()
    assert SEAMLESS_CLONE.code_export is not None
    lines = SEAMLESS_CLONE.code_export(
        {"mode": "normal", "auto_center": True, "center_x": 0, "center_y": 0},
        ("src", "dst", "mask"), "out",
    )
    namespace = {"src": src, "dst": dst, "mask": mask, "cv2": cv2, "np": np}
    exec("\n".join(lines), namespace)
    assert namespace["out"].shape == dst.shape


def test_seamless_clone_code_export_manual_centre_branch() -> None:
    """The codegen branches on auto_center; the manual branch must produce
    a valid runnable program too."""
    import cv2

    src, dst, mask = _seamless_inputs()
    assert SEAMLESS_CLONE.code_export is not None
    lines = SEAMLESS_CLONE.code_export(
        {"mode": "mixed", "auto_center": False, "center_x": 32, "center_y": 32},
        ("src", "dst", "mask"), "out",
    )
    namespace = {"src": src, "dst": dst, "mask": mask, "cv2": cv2, "np": np}
    exec("\n".join(lines), namespace)
    assert namespace["out"].shape == dst.shape
