from __future__ import annotations

import numpy as np
import pytest

from cvstudio.core.operation import OperationSpec, Parameter
from cvstudio.core.pipeline import Pipeline, PipelineNode, Roi


def _add_constant(image: np.ndarray, value: int) -> np.ndarray:
    return np.clip(image.astype(np.int32) + value, 0, 255).astype(np.uint8)


ADD = OperationSpec(
    id="test.add_constant",
    name="Add Constant",
    category="Test",
    description="Adds a constant to every pixel.",
    parameters=(Parameter(name="value", kind="int", default=10, min=-255, max=255),),
    func=_add_constant,
)


def _gray_image() -> np.ndarray:
    return np.full((4, 4, 3), 100, dtype=np.uint8)


def test_empty_pipeline_returns_copy_of_input() -> None:
    pipe = Pipeline()
    img = _gray_image()
    out = pipe.execute(img)
    assert np.array_equal(out, img)
    assert out is not img, "pipeline must not return the input array itself"


def test_pipeline_applies_single_operation() -> None:
    pipe = Pipeline()
    pipe.add(ADD)
    out = pipe.execute(_gray_image())
    assert out[0, 0, 0] == 110


def test_pipeline_applies_operations_in_order() -> None:
    pipe = Pipeline()
    pipe.add(ADD, {"value": 10})
    pipe.add(ADD, {"value": 25})
    out = pipe.execute(_gray_image())
    assert out[0, 0, 0] == 135


def test_pipeline_node_fills_default_params() -> None:
    node = PipelineNode(spec=ADD)
    assert node.params == {"value": 10}


def test_pipeline_node_rejects_unknown_params() -> None:
    with pytest.raises(ValueError, match="Unknown parameter"):
        PipelineNode(spec=ADD, params={"bogus": 1})


def test_disabled_node_is_skipped() -> None:
    pipe = Pipeline()
    node = pipe.add(ADD, {"value": 50})
    node.enabled = False
    out = pipe.execute(_gray_image())
    assert out[0, 0, 0] == 100  # unchanged


def test_pipeline_remove_and_move() -> None:
    pipe = Pipeline()
    a = pipe.add(ADD, {"value": 1})
    b = pipe.add(ADD, {"value": 2})
    c = pipe.add(ADD, {"value": 3})
    pipe.move(0, 2)
    assert pipe.nodes == [b, c, a]
    removed = pipe.remove(1)
    assert removed is c
    assert len(pipe) == 2


def test_pipeline_reorder_permutes_nodes() -> None:
    pipe = Pipeline()
    a = pipe.add(ADD, {"value": 1})
    b = pipe.add(ADD, {"value": 2})
    c = pipe.add(ADD, {"value": 3})
    pipe.reorder([2, 0, 1])
    assert pipe.nodes == [c, a, b]


def test_pipeline_reorder_identity_is_no_op() -> None:
    pipe = Pipeline()
    a = pipe.add(ADD)
    b = pipe.add(ADD)
    pipe.reorder([0, 1])
    assert pipe.nodes == [a, b]


def test_pipeline_reorder_rejects_wrong_length() -> None:
    pipe = Pipeline()
    pipe.add(ADD)
    pipe.add(ADD)
    with pytest.raises(ValueError, match="permutation"):
        pipe.reorder([0])


def test_pipeline_reorder_rejects_non_permutation() -> None:
    pipe = Pipeline()
    pipe.add(ADD)
    pipe.add(ADD)
    with pytest.raises(ValueError, match="permutation"):
        pipe.reorder([0, 0])  # duplicate index — not a permutation


def test_roi_rejects_non_positive_extent() -> None:
    with pytest.raises(ValueError, match="positive"):
        Roi(x=0, y=0, width=0, height=5)
    with pytest.raises(ValueError, match="positive"):
        Roi(x=0, y=0, width=5, height=-1)


def test_roi_clipped_to_bounds_returns_intersection() -> None:
    roi = Roi(x=-10, y=5, width=40, height=20)
    clipped = roi.clipped_to((30, 25))  # image is 25 wide, 30 tall
    assert clipped is not None
    assert (clipped.x, clipped.y, clipped.width, clipped.height) == (0, 5, 25, 20)


def test_roi_clipped_to_outside_image_returns_none() -> None:
    roi = Roi(x=200, y=200, width=50, height=50)
    assert roi.clipped_to((100, 100)) is None


def test_pipeline_with_roi_only_affects_the_selected_region() -> None:
    pipe = Pipeline()
    pipe.add(ADD, {"value": 50})
    pipe.roi = Roi(x=1, y=1, width=2, height=2)
    img = _gray_image()  # all 100s, shape (4, 4, 3)
    out = pipe.execute(img)
    # Outside the ROI the source must be untouched.
    assert int(out[0, 0, 0]) == 100
    assert int(out[3, 3, 0]) == 100
    # Inside, pixels are bumped by 50.
    assert int(out[1, 1, 0]) == 150
    assert int(out[2, 2, 0]) == 150


def test_pipeline_with_partially_off_image_roi_clips_to_bounds() -> None:
    pipe = Pipeline()
    pipe.add(ADD, {"value": 30})
    pipe.roi = Roi(x=-5, y=-5, width=10, height=10)  # only (0,0)-(4,4) overlaps a 4x4 image
    img = _gray_image()
    out = pipe.execute(img)
    # Every pixel in the 4x4 image is inside the clipped ROI → all bumped.
    assert int(out[0, 0, 0]) == 130
    assert int(out[3, 3, 0]) == 130


def test_pipeline_roi_with_no_overlap_returns_source_unchanged() -> None:
    pipe = Pipeline()
    pipe.add(ADD, {"value": 50})
    pipe.roi = Roi(x=100, y=100, width=10, height=10)
    img = _gray_image()
    out = pipe.execute(img)
    assert np.array_equal(out, img)
    assert out is not img  # always a copy


def _resize_half(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    return image[: h // 2, : w // 2]


_RESIZE_SPEC = OperationSpec(
    id="test.resize_half",
    name="Resize Half",
    category="Test",
    description="",
    parameters=(),
    func=_resize_half,
)


def test_pipeline_roi_with_shape_changing_op_returns_source_unchanged() -> None:
    pipe = Pipeline()
    pipe.add(_RESIZE_SPEC)
    pipe.roi = Roi(x=0, y=0, width=2, height=2)
    img = _gray_image()
    out = pipe.execute(img)
    # Splice fails because crop became 1x1 instead of 2x2 — Pipeline falls back to a clean copy.
    assert out.shape == img.shape
    assert np.array_equal(out, img)


def _bgr_to_gray_spec() -> OperationSpec:
    def _convert(image: np.ndarray) -> np.ndarray:
        import cv2

        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    return OperationSpec(
        id="test.to_gray",
        name="To Gray",
        category="Test",
        description="",
        parameters=(),
        func=_convert,
    )


def test_pipeline_roi_handles_grayscale_step_inside_bgr_source() -> None:
    """A To-Grayscale op inside an ROI on a BGR source must not silently no-op.
    The grayscale crop is promoted back to BGR for splice."""
    pipe = Pipeline()
    pipe.add(_bgr_to_gray_spec())
    pipe.roi = Roi(x=1, y=1, width=2, height=2)
    img = _gray_image()  # all (100, 100, 100) BGR
    out = pipe.execute(img)
    assert out.shape == img.shape
    # Inside the ROI a true grayscale conversion of (100,100,100) yields ~100
    # replicated across 3 channels — same value but the splice has actually
    # written to those pixels (no untouched-source fallback).
    assert int(out[1, 1, 0]) == 100


def _channel_zero_then_promote_spec() -> OperationSpec:
    """Returns the blue channel only — a 2D ndarray — so we can verify that
    coercion DOES change the visible result in BGR splice mode."""
    def _zero_red(image: np.ndarray) -> np.ndarray:
        out = image.copy()
        if out.ndim == 3 and out.shape[2] >= 3:
            out[..., 2] = 0  # zero the red channel
        return out

    return OperationSpec(
        id="test.zero_red",
        name="Zero Red",
        category="Test",
        description="",
        parameters=(),
        func=_zero_red,
    )


def test_pipeline_roi_with_channel_modifying_step_writes_to_splice_area() -> None:
    pipe = Pipeline()
    pipe.add(_channel_zero_then_promote_spec())
    pipe.roi = Roi(x=1, y=1, width=2, height=2)
    img = _gray_image()  # all 100 BGR
    out = pipe.execute(img)
    # Outside ROI: red stays at 100. Inside ROI: red dropped to 0.
    assert int(out[0, 0, 2]) == 100
    assert int(out[1, 1, 2]) == 0
    assert int(out[2, 2, 2]) == 0
    assert int(out[3, 3, 2]) == 100


def test_pipeline_clear_also_clears_roi() -> None:
    pipe = Pipeline()
    pipe.add(ADD)
    pipe.roi = Roi(x=0, y=0, width=2, height=2)
    pipe.clear()
    assert pipe.roi is None


def test_pipeline_does_not_mutate_input() -> None:
    pipe = Pipeline()
    pipe.add(ADD, {"value": 50})
    img = _gray_image()
    original = img.copy()
    pipe.execute(img)
    assert np.array_equal(img, original)


# ---------------------------------------------------------------- rotated ROI


def test_roi_defaults_angle_to_zero() -> None:
    roi = Roi(x=0, y=0, width=10, height=10)
    assert roi.angle == 0.0


def test_pipeline_with_zero_angle_roi_matches_axis_aligned_path() -> None:
    """A rotated ROI with angle=0 must produce a bit-identical result to the
    legacy axis-aligned path — guards against regressions in the fast path."""
    img = np.tile(np.arange(8, dtype=np.uint8), (8, 1))[..., None].repeat(3, axis=2)

    pipe_legacy = Pipeline()
    pipe_legacy.add(ADD, {"value": 30})
    pipe_legacy.roi = Roi(x=2, y=2, width=4, height=4)
    out_legacy = pipe_legacy.execute(img)

    pipe_explicit = Pipeline()
    pipe_explicit.add(ADD, {"value": 30})
    pipe_explicit.roi = Roi(x=2, y=2, width=4, height=4, angle=0.0)
    out_explicit = pipe_explicit.execute(img)

    assert np.array_equal(out_legacy, out_explicit)


def test_pipeline_with_rotated_roi_affects_only_rotated_region() -> None:
    """On a uniform image, a 45° ROI must bump pixels inside the rotated quad
    and leave the four image corners (which fall outside the rotated rect)
    completely untouched."""
    img = np.full((40, 40, 3), 100, dtype=np.uint8)
    pipe = Pipeline()
    pipe.add(ADD, {"value": 80})
    # A 20x20 rect centred on (20, 20), rotated 45°, has corners pointing
    # along the axes — its bounding box reaches (~6, 20) on the left and
    # (~34, 20) on the right, but the image corners (0,0), (0,39), (39,0),
    # (39,39) remain well outside.
    pipe.roi = Roi(x=10, y=10, width=20, height=20, angle=45.0)
    out = pipe.execute(img)
    # Corners untouched.
    assert int(out[0, 0, 0]) == 100
    assert int(out[0, 39, 0]) == 100
    assert int(out[39, 0, 0]) == 100
    assert int(out[39, 39, 0]) == 100
    # Centre is inside the rotated quad → bumped.
    assert int(out[20, 20, 0]) == 180


def test_pipeline_rotated_roi_with_paste_to_relocates_processed_patch() -> None:
    """paste_to + rotation deposits the rotated patch at the new centre, with
    the source pixels at the ORIGINAL ROI location left unchanged."""
    img = np.full((60, 60, 3), 50, dtype=np.uint8)
    pipe = Pipeline()
    pipe.add(ADD, {"value": 100})
    pipe.roi = Roi(x=5, y=5, width=10, height=10, angle=30.0)
    pipe.roi_paste_to = (40, 40)
    out = pipe.execute(img)
    # Original ROI centre — untouched (the rotated patch landed elsewhere).
    assert int(out[10, 10, 0]) == 50
    # Paste-to centre (x=45, y=45) — inside the rotated patch → bumped.
    assert int(out[45, 45, 0]) == 150


def test_pipeline_rotated_roi_with_shape_changing_op_returns_source_unchanged() -> None:
    """Same fallback contract as the axis-aligned path: a shape-changing op
    inside a rotated ROI must not raise — Pipeline returns a clean copy."""
    pipe = Pipeline()
    pipe.add(_RESIZE_SPEC)
    pipe.roi = Roi(x=0, y=0, width=4, height=4, angle=20.0)
    img = _gray_image()
    out = pipe.execute(img)
    assert out.shape == img.shape
    assert np.array_equal(out, img)


# ----------------------------------------------------- needs_source kwarg


def _make_needs_source_spec(
    captured: dict,
) -> OperationSpec:
    """Build a one-off OperationSpec that records the kwargs Graph.execute
    forwarded to it. Used by the next two tests."""

    def _record(image: np.ndarray, **kwargs) -> np.ndarray:
        captured["image_shape"] = image.shape
        captured["kwargs_keys"] = sorted(kwargs.keys())
        captured["source_image_shape"] = (
            kwargs["_source_image"].shape if "_source_image" in kwargs else None
        )
        captured["roi_origin"] = kwargs.get("_roi_origin")
        return image  # pass-through; we only care about what was injected

    return OperationSpec(
        id="test.needs_source_probe",
        name="Needs Source Probe",
        category="Test",
        description="",
        parameters=(),
        func=_record,
        needs_source=True,
    )


def test_pipeline_execute_passes_source_image_to_needs_source_ops() -> None:
    """When a Pipeline ROI is set and the node spec has needs_source=True,
    Graph.execute forwards the full pre-ROI source image plus the ROI
    origin as _source_image / _roi_origin kwargs."""
    captured: dict = {}
    pipe = Pipeline()
    pipe.add(_make_needs_source_spec(captured))
    pipe.roi = Roi(x=1, y=1, width=2, height=2)
    img = _gray_image()  # 4x4x3 of 100s
    pipe.execute(img)
    assert captured["image_shape"] == (2, 2, 3)  # the cropped ROI
    assert "_source_image" in captured["kwargs_keys"]
    assert "_roi_origin" in captured["kwargs_keys"]
    assert captured["source_image_shape"] == img.shape
    assert captured["roi_origin"] == (1, 1)


def test_pipeline_execute_does_not_inject_kwargs_into_normal_ops() -> None:
    """Regression guard: ops without needs_source MUST NOT see the
    _source_image / _roi_origin kwargs. Adding unexpected kwargs would
    break every existing op's signature."""
    captured: dict = {}

    def _record(image: np.ndarray, **kwargs) -> np.ndarray:
        captured["kwargs_keys"] = sorted(kwargs.keys())
        return image

    spec = OperationSpec(
        id="test.no_needs_source",
        name="No Needs Source",
        category="Test",
        description="",
        parameters=(),
        func=_record,
        needs_source=False,
    )
    pipe = Pipeline()
    pipe.add(spec)
    pipe.roi = Roi(x=1, y=1, width=2, height=2)
    pipe.execute(_gray_image())
    assert "_source_image" not in captured["kwargs_keys"]
    assert "_roi_origin" not in captured["kwargs_keys"]
