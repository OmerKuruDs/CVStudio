from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from cvstudio.core.batch import (
    BatchRequest,
    discover_images,
    execute_batch,
    make_request,
)


def _write_test_image(path: Path, value: int = 50) -> None:
    img = np.full((4, 4, 3), value, dtype=np.uint8)
    cv2.imwrite(str(path), img)


def _add_50(image: np.ndarray) -> np.ndarray:
    return np.clip(image.astype(np.int32) + 50, 0, 255).astype(np.uint8)


def test_discover_images_returns_supported_extensions_only(tmp_path: Path) -> None:
    _write_test_image(tmp_path / "a.png")
    _write_test_image(tmp_path / "b.jpg")
    (tmp_path / "notes.txt").write_text("ignored", encoding="utf-8")
    (tmp_path / "data.npy").write_bytes(b"\x00\x00")
    found = discover_images(tmp_path)
    assert [p.name for p in found] == ["a.png", "b.jpg"]


def test_discover_images_returns_empty_for_nonexistent_dir(tmp_path: Path) -> None:
    assert discover_images(tmp_path / "missing") == ()


def test_execute_batch_processes_each_image_and_writes_output(tmp_path: Path) -> None:
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    for name in ("a.png", "b.png"):
        _write_test_image(input_dir / name, value=50)

    request = make_request(
        input_dir=input_dir,
        output_dir=output_dir,
        process=_add_50,
        output_suffix="_done",
    )
    result = execute_batch(request)
    assert result.completed == 2
    assert result.failures == ()

    written = cv2.imread(str(output_dir / "a_done.png"))
    assert int(written[0, 0, 0]) == 100  # +50 applied


def test_execute_batch_skips_existing_when_not_overwriting(tmp_path: Path) -> None:
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()
    _write_test_image(input_dir / "a.png", value=10)
    _write_test_image(output_dir / "a_done.png", value=200)  # pre-existing

    request = make_request(
        input_dir=input_dir,
        output_dir=output_dir,
        process=_add_50,
        output_suffix="_done",
    )
    result = execute_batch(request)
    assert result.completed == 0
    assert result.skipped == 1
    # The pre-existing file is untouched.
    assert int(cv2.imread(str(output_dir / "a_done.png"))[0, 0, 0]) == 200


def test_execute_batch_overwrites_when_flag_is_set(tmp_path: Path) -> None:
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()
    _write_test_image(input_dir / "a.png", value=10)
    _write_test_image(output_dir / "a_done.png", value=200)

    request = make_request(
        input_dir=input_dir,
        output_dir=output_dir,
        process=_add_50,
        output_suffix="_done",
        overwrite=True,
    )
    result = execute_batch(request)
    assert result.completed == 1
    # Now overwritten with the pipeline output (10 + 50 = 60).
    assert int(cv2.imread(str(output_dir / "a_done.png"))[0, 0, 0]) == 60


def test_execute_batch_records_failures_but_continues(tmp_path: Path) -> None:
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    _write_test_image(input_dir / "ok.png", value=10)
    (input_dir / "bad.png").write_bytes(b"not really an image")
    _write_test_image(input_dir / "ok2.png", value=20)

    request = make_request(
        input_dir=input_dir,
        output_dir=output_dir,
        process=_add_50,
        output_suffix="_done",
    )
    result = execute_batch(request)
    assert result.completed == 2
    assert len(result.failures) == 1
    failed_path, _ = result.failures[0]
    assert failed_path.name == "bad.png"


def test_execute_batch_honours_cancellation_between_files(tmp_path: Path) -> None:
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    for i in range(5):
        _write_test_image(input_dir / f"img_{i}.png", value=i)

    seen: list[int] = []

    def process(img: np.ndarray) -> np.ndarray:
        seen.append(int(img[0, 0, 0]))
        return img

    # Cancel after the second file by checking call count.
    state = {"count": 0}

    def cancel_check() -> bool:
        state["count"] += 1
        return state["count"] > 2  # allow first two iterations to start

    request = make_request(
        input_dir=input_dir,
        output_dir=output_dir,
        process=process,
        output_suffix="_done",
    )
    result = execute_batch(request, cancel_check=cancel_check)
    assert result.cancelled is True
    assert result.completed < 5


def test_execute_batch_emits_progress_for_each_file(tmp_path: Path) -> None:
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    for name in ("a.png", "b.png", "c.png"):
        _write_test_image(input_dir / name, value=10)

    events: list[tuple[int, int, str]] = []

    def on_progress(done: int, total: int, path: Path) -> None:
        events.append((done, total, path.name))

    request = make_request(
        input_dir=input_dir,
        output_dir=output_dir,
        process=_add_50,
        output_suffix="_p",
    )
    execute_batch(request, on_progress=on_progress)
    # 3 per-file ticks + 1 final tick.
    assert len(events) == 4
    assert events[-1][0] == 3 and events[-1][1] == 3


def test_batch_request_output_path_uses_suffix_and_keeps_extension() -> None:
    request = BatchRequest(
        input_paths=(),
        output_dir=Path("/out"),
        process=_add_50,
        output_suffix="_clean",
    )
    assert request.output_path_for(Path("photo.PNG")) == Path("/out/photo_clean.PNG")


def test_batch_request_variant_output_path_inserts_variant_suffix() -> None:
    request = BatchRequest(
        input_paths=(),
        output_dir=Path("/out"),
        process=_add_50,
        output_suffix="_synth",
        variants_per_image=3,
    )
    assert request.output_path_for(Path("p.png"), variant_index=0) == Path("/out/p_synth_v0.png")
    assert request.output_path_for(Path("p.png"), variant_index=2) == Path("/out/p_synth_v2.png")


def test_execute_batch_writes_n_variants_per_input(tmp_path: Path) -> None:
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    _write_test_image(input_dir / "a.png", value=10)
    _write_test_image(input_dir / "b.png", value=20)

    seen_indices: list[int] = []

    def process(image: np.ndarray, variant_index: int) -> np.ndarray:
        seen_indices.append(variant_index)
        # Encode the variant index into pixel 0 so each output file is distinct.
        out = image.copy()
        out[0, 0] = variant_index
        return out

    request = BatchRequest(
        input_paths=tuple(sorted(input_dir.iterdir())),
        output_dir=output_dir,
        process=process,
        output_suffix="_v",
        variants_per_image=3,
    )
    result = execute_batch(request)
    assert result.completed == 6
    assert result.failures == ()
    # Each input got every variant index.
    assert seen_indices == [0, 1, 2, 0, 1, 2]
    # Each variant's pixel 0 carries its index, proving the files differ.
    for name in ("a", "b"):
        for v in range(3):
            written = cv2.imread(str(output_dir / f"{name}_v_v{v}.png"))
            assert int(written[0, 0, 0]) == v


def test_execute_batch_falls_back_to_single_arg_process_when_no_variant_index(tmp_path: Path) -> None:
    """Old callers pass `process(image) -> image`; the executor must keep
    working with them when variants_per_image is 1 (the default)."""
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    _write_test_image(input_dir / "a.png", value=30)

    request = BatchRequest(
        input_paths=(input_dir / "a.png",),
        output_dir=output_dir,
        process=_add_50,           # single-arg signature
        output_suffix="_done",
    )
    result = execute_batch(request)
    assert result.completed == 1
    assert result.failures == ()


def test_execute_batch_multi_variant_emits_per_unit_progress(tmp_path: Path) -> None:
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    _write_test_image(input_dir / "a.png", value=10)
    _write_test_image(input_dir / "b.png", value=20)

    events: list[tuple[int, int]] = []

    def on_progress(done: int, total: int, _path: Path) -> None:
        events.append((done, total))

    request = BatchRequest(
        input_paths=tuple(sorted(input_dir.iterdir())),
        output_dir=output_dir,
        process=_add_50,
        output_suffix="_p",
        variants_per_image=2,
    )
    execute_batch(request, on_progress=on_progress)
    # 2 inputs × 2 variants = 4 per-unit ticks plus 1 final tick.
    assert len(events) == 5
    assert events[-1] == (4, 4)


@pytest.fixture
def temp_pipeline_callable():
    return _add_50
