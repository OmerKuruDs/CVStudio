from __future__ import annotations

import cv2
import numpy as np
import pytest

from cvstudio.defects import GrainFlatteningLabel, inject_grain_flattening


# --------------------------------------------------------------- test images


def _grainy_image(h: int = 200, w: int = 200, seed: int = 0) -> np.ndarray:
    """A pseudo grain-microscope image: low-frequency noise that has lots of
    high-contrast edges Canny can latch onto, so the auto-ROI sampler has
    work to do and the directional smear has texture to chew on."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, size=(h, w), dtype=np.uint8)
    # Slight blur produces grain-like blobs instead of pure white noise.
    return cv2.GaussianBlur(base, (5, 5), 0)


def _grainy_color_image(h: int = 200, w: int = 200, seed: int = 0) -> np.ndarray:
    gray = _grainy_image(h, w, seed)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# ----------------------------------------------------------------- happy path


def test_output_has_same_shape_and_dtype_as_input() -> None:
    img = _grainy_image()
    out, label = inject_grain_flattening(img, intensity=0.5, seed=0)
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    assert isinstance(label, GrainFlatteningLabel)


def test_roi_region_actually_changes() -> None:
    img = _grainy_image()
    roi = (50, 50, 60, 60)
    out, _ = inject_grain_flattening(img, roi_box=roi, intensity=0.7, seed=0)
    # The patched ROI must differ from the original — otherwise the
    # injection silently produced a no-op.
    inside = out[50:110, 50:110]
    original_inside = img[50:110, 50:110]
    assert not np.array_equal(inside, original_inside)


def test_pixels_far_outside_roi_are_left_untouched() -> None:
    img = _grainy_image()
    roi = (60, 60, 40, 40)  # tight ROI; feather fringe stays well inside (60,100)
    out, _ = inject_grain_flattening(
        img, roi_box=roi, intensity=0.5, feather_fraction=0.2, seed=0
    )
    # Corners are far from the ROI + its feather fringe → must be untouched.
    assert int(out[0, 0]) == int(img[0, 0])
    assert int(out[199, 199]) == int(img[199, 199])


def test_color_input_returns_color_output() -> None:
    img = _grainy_color_image()
    out, label = inject_grain_flattening(img, intensity=0.5, seed=0)
    assert out.shape == img.shape
    assert out.shape[2] == 3
    # Label image_shape is (H, W), independent of channels.
    assert label.image_shape == img.shape[:2]


# ---------------------------------------------------------------- direction


def test_horizontal_vs_vertical_produce_different_results() -> None:
    img = _grainy_image()
    roi = (50, 50, 60, 60)
    out_h, _ = inject_grain_flattening(
        img, roi_box=roi, flattening_direction="horizontal", intensity=0.8, seed=0
    )
    out_v, _ = inject_grain_flattening(
        img, roi_box=roi, flattening_direction="vertical", intensity=0.8, seed=0
    )
    assert not np.array_equal(out_h, out_v)


def test_horizontal_squeeze_raises_row_variation_above_column_variation() -> None:
    """Horizontal squeeze packs the same grain texture into a narrower
    width and then mirror-tiles it — so adjacent pixels along a row sweep
    through detail faster than adjacent pixels in a column. The ROI
    interior should therefore show higher mean row-to-row delta than
    column-to-column delta. Measured in the inner fully-opaque region of
    the alpha mask so the feather fringe doesn't dilute the effect."""
    img = _grainy_image(seed=1)
    roi = (40, 40, 120, 120)
    out, _ = inject_grain_flattening(
        img, roi_box=roi,
        flattening_direction="horizontal", intensity=1.0,
        feather_fraction=0.05, seed=0,
    )
    # Sample the centre 60x60 — well inside the ROI, far from any fringe.
    patch = out[70:130, 70:130].astype(np.int32)
    row_diff = float(np.abs(np.diff(patch, axis=1)).mean())
    col_diff = float(np.abs(np.diff(patch, axis=0)).mean())
    assert row_diff > col_diff


# ----------------------------------------------------------------- intensity


def test_intensity_zero_produces_smaller_change_than_intensity_one() -> None:
    img = _grainy_image()
    roi = (50, 50, 60, 60)
    # Small feather so the ROI interior reflects intensity changes instead
    # of being dominated by the blend with the untouched original.
    out_low, _ = inject_grain_flattening(
        img, roi_box=roi, intensity=0.0, feather_fraction=0.05, seed=0
    )
    out_high, _ = inject_grain_flattening(
        img, roi_box=roi, intensity=1.0, feather_fraction=0.05, seed=0
    )
    inside = (slice(55, 105), slice(55, 105))
    diff_low = float(np.abs(out_low[inside].astype(np.int32) - img[inside].astype(np.int32)).mean())
    diff_high = float(np.abs(out_high[inside].astype(np.int32) - img[inside].astype(np.int32)).mean())
    # High intensity must dominate by a multiplicative margin — absolute
    # thresholds depend on input statistics and aren't portable.
    assert diff_high > diff_low * 2.0


# ----------------------------------------------------------------- auto ROI


def test_auto_roi_picks_a_box_inside_the_image() -> None:
    img = _grainy_image()
    _, label = inject_grain_flattening(img, intensity=0.5, seed=42)
    x, y, w, h = label.bbox_xywh
    assert x >= 0 and y >= 0
    assert x + w <= img.shape[1]
    assert y + h <= img.shape[0]
    assert w > 0 and h > 0


def test_auto_roi_is_reproducible_with_same_seed() -> None:
    img = _grainy_image()
    _, label_a = inject_grain_flattening(img, intensity=0.5, seed=7)
    _, label_b = inject_grain_flattening(img, intensity=0.5, seed=7)
    assert label_a.bbox_xywh == label_b.bbox_xywh


def test_auto_roi_different_seeds_give_different_boxes() -> None:
    img = _grainy_image()
    _, label_a = inject_grain_flattening(img, intensity=0.5, seed=1)
    _, label_b = inject_grain_flattening(img, intensity=0.5, seed=2)
    assert label_a.bbox_xywh != label_b.bbox_xywh


def test_auto_roi_falls_back_to_centre_on_flat_image() -> None:
    """Canny finds no edges on a uniform image — the sampler must still
    return a sensible centred ROI rather than crash."""
    flat = np.full((120, 120), 128, dtype=np.uint8)
    out, label = inject_grain_flattening(flat, intensity=0.5, seed=0)
    assert out.shape == flat.shape
    x, y, w, h = label.bbox_xywh
    assert 0 <= x and 0 <= y
    assert x + w <= flat.shape[1]
    assert y + h <= flat.shape[0]


# ------------------------------------------------------------------- labels


def test_label_bbox_fields_are_consistent() -> None:
    img = _grainy_image()
    roi = (30, 40, 50, 60)
    _, label = inject_grain_flattening(img, roi_box=roi, seed=0)
    x, y, w, h = label.bbox_xywh
    assert label.bbox_xyxy == (x, y, x + w, y + h)
    # Polygon = 4 closed rectangle corners.
    assert label.polygon == ((x, y), (x + w, y), (x + w, y + h), (x, y + h))


def test_label_yolo_values_are_normalised_to_unit_interval() -> None:
    img = _grainy_image(h=200, w=300)
    roi = (60, 80, 100, 40)
    _, label = inject_grain_flattening(img, roi_box=roi, seed=0)
    cx, cy, w, h = label.yolo
    for value in (cx, cy, w, h):
        assert 0.0 <= value <= 1.0
    # Spot-check: centre x = (60 + 100/2) / 300 = 110 / 300
    assert abs(cx - 110 / 300) < 1e-6
    # width = 100 / 300
    assert abs(w - 100 / 300) < 1e-6


def test_label_records_direction_and_intensity() -> None:
    img = _grainy_image()
    _, label = inject_grain_flattening(
        img, roi_box=(10, 10, 30, 30),
        flattening_direction="vertical", intensity=0.42, seed=0,
    )
    assert label.direction == "vertical"
    assert label.angle_degrees == pytest.approx(90.0)
    assert label.intensity == pytest.approx(0.42)


def test_angle_degrees_overrides_flattening_direction() -> None:
    img = _grainy_image()
    _, label = inject_grain_flattening(
        img, roi_box=(10, 10, 30, 30),
        flattening_direction="horizontal", angle_degrees=45.0, seed=0,
    )
    # angle wins; direction string stays as the user-supplied legacy tag.
    assert label.angle_degrees == pytest.approx(45.0)


def test_angle_degrees_wraps_into_half_circle() -> None:
    img = _grainy_image()
    for raw, expected in [(180.0, 0.0), (270.0, 90.0), (-45.0, 135.0)]:
        _, label = inject_grain_flattening(
            img, roi_box=(10, 10, 30, 30), angle_degrees=raw, seed=0,
        )
        assert label.angle_degrees == pytest.approx(expected)


def test_angled_smear_produces_different_output_than_horizontal() -> None:
    img = _grainy_image(seed=2)
    roi = (40, 40, 120, 120)
    out_h, _ = inject_grain_flattening(
        img, roi_box=roi, angle_degrees=0.0,
        intensity=1.0, feather_fraction=0.05, seed=0,
    )
    out_45, _ = inject_grain_flattening(
        img, roi_box=roi, angle_degrees=45.0,
        intensity=1.0, feather_fraction=0.05, seed=0,
    )
    assert not np.array_equal(out_h, out_45)


def test_fill_roi_treats_whole_image_as_defect_region() -> None:
    img = _grainy_image(h=80, w=80)
    out, label = inject_grain_flattening(
        img, fill_roi=True, intensity=0.6, feather_fraction=0.1, seed=0,
    )
    # Bbox covers the whole image — fill_roi bypasses sub-ROI sampling.
    assert label.bbox_xywh == (0, 0, 80, 80)
    # And the output really did change (the centre is well inside the
    # alpha-mask plateau even with a small feather).
    assert not np.array_equal(out, img)


def test_fill_roi_ignores_explicit_roi_box() -> None:
    img = _grainy_image(h=80, w=80)
    _, label = inject_grain_flattening(
        img, fill_roi=True, roi_box=(10, 10, 20, 20), intensity=0.5, seed=0,
    )
    # fill_roi takes precedence — roi_box is silently ignored.
    assert label.bbox_xywh == (0, 0, 80, 80)


# -------------------------------------------------------------- roi clamping


def test_roi_clamped_to_image_bounds_when_partially_off_screen() -> None:
    img = _grainy_image(h=100, w=100)
    # ROI extends past the right edge — must be clipped, not raise.
    out, label = inject_grain_flattening(
        img, roi_box=(80, 80, 50, 50), intensity=0.5, seed=0
    )
    assert out.shape == img.shape
    x, y, w, h = label.bbox_xywh
    assert x + w <= img.shape[1]
    assert y + h <= img.shape[0]


def test_roi_completely_off_screen_raises_value_error() -> None:
    img = _grainy_image(h=100, w=100)
    with pytest.raises(ValueError, match="no overlap"):
        inject_grain_flattening(img, roi_box=(200, 200, 50, 50), intensity=0.5)


# --------------------------------------------------------------- validation


def test_rejects_non_uint8_image() -> None:
    img = np.zeros((50, 50), dtype=np.float32)
    with pytest.raises(ValueError, match="uint8"):
        inject_grain_flattening(img)  # type: ignore[arg-type]


def test_rejects_unknown_direction() -> None:
    img = _grainy_image()
    with pytest.raises(ValueError, match="flattening_direction"):
        inject_grain_flattening(img, flattening_direction="diagonal")  # type: ignore[arg-type]


def test_rejects_intensity_out_of_range() -> None:
    img = _grainy_image()
    with pytest.raises(ValueError, match="intensity"):
        inject_grain_flattening(img, intensity=1.5)
    with pytest.raises(ValueError, match="intensity"):
        inject_grain_flattening(img, intensity=-0.1)


def test_rejects_non_array_input() -> None:
    with pytest.raises(TypeError, match="numpy array"):
        inject_grain_flattening("not an image")  # type: ignore[arg-type]


def test_rejects_4d_input() -> None:
    img = np.zeros((10, 10, 3, 1), dtype=np.uint8)
    with pytest.raises(ValueError, match="ndim"):
        inject_grain_flattening(img)


# --------------------------------------------------------------- feathering


# ------------------------------------------------------- geometric squeeze


def test_squeeze_preserves_high_frequency_detail() -> None:
    """The whole point of the geometric squeeze: grain micro-detail must
    survive. We measure high-frequency content with Laplacian variance —
    a classic sharpness proxy. The squeezed ROI should keep at least 50 %
    of the original's Laplacian variance. A pure Gaussian blur or motion
    blur over the same patch would typically drop this below 20 %."""
    img = _grainy_image(seed=3)
    roi = (40, 40, 120, 120)
    inside = (slice(70, 130), slice(70, 130))
    original_lap_var = float(cv2.Laplacian(img[inside], cv2.CV_64F).var())

    out, _ = inject_grain_flattening(
        img, roi_box=roi, intensity=1.0, feather_fraction=0.0, seed=0,
    )
    squeezed_lap_var = float(cv2.Laplacian(out[inside], cv2.CV_64F).var())

    assert squeezed_lap_var > original_lap_var * 0.5, (
        f"high-frequency detail lost: original lap_var={original_lap_var:.1f}, "
        f"squeezed lap_var={squeezed_lap_var:.1f}"
    )


def test_lanczos_squeeze_does_not_smear_a_strong_edge() -> None:
    """A black-to-white step edge inside the ROI must stay a near-step
    after squeezing. We check the transition width — the count of pixels
    between the dark and bright extremes — stays small. A blur algorithm
    would spread this across many pixels."""
    img = np.zeros((120, 120), dtype=np.uint8)
    img[:, 60:] = 255  # vertical step edge in the middle
    roi = (20, 20, 80, 80)
    out, _ = inject_grain_flattening(
        img, roi_box=roi, intensity=0.5, feather_fraction=0.0,
        flattening_direction="vertical",  # squeeze vertically — leaves the horizontal step in place
        seed=0,
    )
    # Sample a single row through the ROI centre; count the transition
    # band — pixels strictly between 30 and 225.
    centre_row = out[60, 30:90]
    transition_width = int(np.sum((centre_row > 30) & (centre_row < 225)))
    assert transition_width <= 3, (
        f"LANCZOS4 squeeze blurred the step edge across {transition_width} "
        f"pixels; expected ≤ 3"
    )


def test_no_centre_mirror_butterfly_artifact() -> None:
    """The previous mirror-tile algorithm produced a Rorschach-style
    butterfly: the ROI's left half was a near-perfect reflection of its
    right half. A downstream model would learn this in seconds. The new
    wide-context squeeze must not reproduce that symmetry — compare the
    left half of the ROI with its horizontally-flipped right half and
    require a non-trivial mean absolute difference."""
    img = _grainy_image(seed=5)
    roi_x, roi_y, roi_w, roi_h = 60, 60, 120, 80
    out, _ = inject_grain_flattening(
        img, roi_box=(roi_x, roi_y, roi_w, roi_h),
        intensity=1.0, feather_fraction=0.0, seed=0,
    )
    inside = out[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w].astype(np.int32)
    half = roi_w // 2
    left = inside[:, :half]
    right_flipped = inside[:, -half:][:, ::-1]
    mirror_diff = float(np.abs(left - right_flipped).mean())
    # Pure mirror tiling drives this below ~1; natural asymmetric texture
    # sits well above 5. We require > 4 as a safe floor.
    assert mirror_diff > 4.0, (
        f"left/right halves are mirror-symmetric (mean abs diff "
        f"{mirror_diff:.2f}) — wide-context squeeze did not break symmetry"
    )


def test_wide_context_path_pulls_from_outside_the_roi() -> None:
    """The wide-context squeeze should sample pixels from outside the
    declared ROI when room is available. Easiest way to prove this:
    plant a sharply distinct stripe IMMEDIATELY to the left of the ROI
    (inside the wider context span the algorithm carves out at
    intensity=1.0, which is ~3.3× the ROI width centred on the ROI), set
    the ROI itself to a uniform mid-grey, and check the squeezed ROI now
    contains the stripe's bright value — only possible via the wider
    crop."""
    img = np.full((100, 400), 128, dtype=np.uint8)
    # ROI 60 px wide at x=220 → centre (220, 250). At ratio=0.3 the
    # wide context span is ~200 px wide, balanced around the ROI centre:
    # [150, 350]. Plant a bright stripe at [155, 215] — well inside the
    # left half of that span but OUTSIDE the ROI.
    img[:, 155:215] = 240
    roi_box = (220, 20, 60, 60)
    # Sanity: the ROI itself contains only the 128-grey base.
    assert int(img[40, 250]) == 128
    out, _ = inject_grain_flattening(
        img, roi_box=roi_box, intensity=1.0, feather_fraction=0.0, seed=0,
    )
    inside = out[20:80, 220:280]
    # Only the wide-context path can pull pixels from outside the ROI;
    # a ROI-only crop is uniformly 128, so any pixel above 200 is proof
    # the bright stripe was sampled.
    assert int(inside.max()) > 200, (
        f"wide-context path did not sample the off-ROI bright stripe "
        f"(max inside={int(inside.max())}); algorithm likely fell back to "
        f"wrap+roll on a context-rich source"
    )


# --------------------------------------------------------- op wrapper paths


def test_grain_flattening_op_uses_source_image_when_provided() -> None:
    """The synthesis op wrapper accepts ``_source_image`` / ``_roi_origin``
    kwargs that Pipeline pushes when needs_source=True. With those args
    the op pulls natural context from outside the cropped ROI — exactly
    the path that used to fall back to the periodic-tile artefact under
    fill_roi=True."""
    from cvstudio.operations.synthesis import _grain_flattening_op  # noqa: PLC0415

    source = np.full((100, 400), 128, dtype=np.uint8)
    source[:, 155:215] = 240  # bright stripe outside the ROI
    roi_origin = (220, 20)
    crop = source[20:80, 220:280].copy()  # what Pipeline would hand the op

    out = _grain_flattening_op(
        crop,
        angle_degrees=0.0,
        intensity=1.0,
        feather_fraction=0.0,
        fill_roi=False,
        seed=0,
        _source_image=source,
        _roi_origin=roi_origin,
    )
    assert out.shape == crop.shape
    # Wide-context squeeze pulled the bright stripe into the ROI.
    assert int(out.max()) > 200, (
        f"op did not use _source_image — max inside ROI={int(out.max())}"
    )


def test_grain_flattening_op_falls_back_to_crop_when_source_missing() -> None:
    """Backward compat: when ``_source_image`` is None the wrapper
    behaves exactly like the standalone library call. Output must still
    have the same shape and dtype as the input crop."""
    from cvstudio.operations.synthesis import _grain_flattening_op  # noqa: PLC0415

    crop = _grainy_image(h=80, w=80, seed=11)
    out = _grain_flattening_op(
        crop,
        angle_degrees=0.0,
        intensity=0.6,
        feather_fraction=0.1,
        fill_roi=True,
        seed=0,
    )
    assert out.shape == crop.shape
    assert out.dtype == crop.dtype
    # And it actually did something.
    assert not np.array_equal(out, crop)


def test_smaller_feather_fraction_leaves_a_more_visible_seam() -> None:
    """A near-zero feather makes the ROI boundary much sharper — the
    one-pixel-outside-ROI difference jumps with the inside-ROI difference,
    not the smooth dissolve a larger feather would produce."""
    img = _grainy_image()
    roi = (60, 60, 40, 40)
    out_soft, _ = inject_grain_flattening(
        img, roi_box=roi, intensity=0.8, feather_fraction=0.45, seed=0
    )
    out_hard, _ = inject_grain_flattening(
        img, roi_box=roi, intensity=0.8, feather_fraction=0.0, seed=0
    )
    # Centre pixel: hard-feather lets the matted texture dominate; soft
    # feather still has most of the original mixed in. So hard ROI shows a
    # larger absolute difference from original at the centre.
    centre = (80, 80)
    hard_centre_delta = abs(int(out_hard[centre]) - int(img[centre]))
    soft_centre_delta = abs(int(out_soft[centre]) - int(img[centre]))
    assert hard_centre_delta >= soft_centre_delta
