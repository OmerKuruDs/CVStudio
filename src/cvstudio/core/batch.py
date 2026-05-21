"""Batch processing — apply a pipeline to every image in a folder.

This module stays Qt-free so it can be unit-tested without spinning up the UI.
The UI layer wraps `execute_batch` in a QThread / QObject so progress and
cancellation hook through Qt signals; here we just take plain callbacks.

A typical request:
    request = BatchRequest(
        input_paths=(...),
        output_dir=Path(...),
        process=pipeline.execute,
        output_suffix="_processed",
        overwrite=False,
    )
    result = execute_batch(request, on_progress=print, cancel_check=...)
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from cvstudio.core.image_io import read_image

IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
)

ProcessFunc = Callable[..., np.ndarray]
"""``(image) -> image`` or ``(image, variant_index) -> image``. The executor
introspects the signature and calls the richer form only when the callable
declares at least two positional parameters."""
ProgressCallback = Callable[[int, int, Path], None]
CancelCheck = Callable[[], bool]


@dataclass(frozen=True)
class BatchRequest:
    input_paths: tuple[Path, ...]
    output_dir: Path
    process: ProcessFunc
    output_suffix: str = ""
    overwrite: bool = False
    variants_per_image: int = 1
    """How many output variants to produce per input image. >1 only makes
    sense if `process` is stochastic and reads seeds from somewhere — the
    typical pattern is to deep-copy the pipeline per variant and override
    every stochastic op's `seed` param with the variant index. When 1 the
    output filename keeps its historic form (no variant suffix)."""

    def output_path_for(self, input_path: Path, variant_index: int = 0) -> Path:
        """Output file for `input_path` at `variant_index` (0-based).

        Single-variant mode (`variants_per_image == 1`) preserves the legacy
        ``{stem}{suffix}{ext}`` shape so existing batches keep round-tripping.
        Multi-variant mode appends ``_v{index}`` after the stem so the
        per-image variants do not collide on disk."""
        if self.variants_per_image <= 1:
            return self.output_dir / f"{input_path.stem}{self.output_suffix}{input_path.suffix}"
        return (
            self.output_dir
            / f"{input_path.stem}{self.output_suffix}_v{variant_index}{input_path.suffix}"
        )


@dataclass(frozen=True)
class BatchResult:
    completed: int
    skipped: int
    failures: tuple[tuple[Path, str], ...]
    cancelled: bool = False


@dataclass
class _Accumulator:
    completed: int = 0
    skipped: int = 0
    failures: list[tuple[Path, str]] = field(default_factory=list)
    cancelled: bool = False


def discover_images(directory: Path) -> tuple[Path, ...]:
    """List every readable image-extension file directly inside `directory`."""
    if not directory.is_dir():
        return ()
    files = [
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    files.sort()
    return tuple(files)


VariantProcessFunc = Callable[[np.ndarray, int], np.ndarray]
"""``(image, variant_index) -> image``. Optional richer alternative to
``ProcessFunc`` so callers that want per-variant determinism can branch
on the index without smuggling state through closures."""


def execute_batch(
    request: BatchRequest,
    *,
    on_progress: ProgressCallback | None = None,
    cancel_check: CancelCheck | None = None,
) -> BatchResult:
    request.output_dir.mkdir(parents=True, exist_ok=True)
    variants = max(1, int(request.variants_per_image))
    total_units = len(request.input_paths) * variants
    acc = _Accumulator()
    accepts_variant = _process_accepts_variant_index(request.process)

    unit_index = 0
    for input_path in request.input_paths:
        if cancel_check is not None and cancel_check():
            acc.cancelled = True
            break

        image: np.ndarray | None = None
        load_error: str | None = None
        try:
            loaded = read_image(input_path)
            if loaded is None:
                load_error = f"unreadable image: {input_path}"
            else:
                image = loaded
        except OSError as exc:
            load_error = str(exc)

        for variant_index in range(variants):
            if cancel_check is not None and cancel_check():
                acc.cancelled = True
                break

            if on_progress is not None:
                on_progress(unit_index, total_units, input_path)
            unit_index += 1

            if load_error is not None:
                acc.failures.append((input_path, load_error))
                continue

            output_path = request.output_path_for(input_path, variant_index=variant_index)
            if output_path.exists() and not request.overwrite:
                acc.skipped += 1
                continue

            try:
                assert image is not None  # narrowed above when load_error is None
                result = (
                    request.process(image, variant_index)  # type: ignore[call-arg]
                    if accepts_variant
                    else request.process(image)
                )
                if not isinstance(result, np.ndarray):
                    raise TypeError(
                        f"pipeline returned {type(result).__name__}, expected ndarray"
                    )
                ok = cv2.imwrite(str(output_path), result)
                if not ok:
                    raise OSError(f"cv2.imwrite returned False for {output_path}")
            except (OSError, TypeError, ValueError) as exc:
                acc.failures.append((input_path, str(exc)))
                continue
            acc.completed += 1

        if acc.cancelled:
            break

    if on_progress is not None and not acc.cancelled:
        on_progress(
            total_units, total_units,
            request.input_paths[-1] if request.input_paths else Path(),
        )

    return BatchResult(
        completed=acc.completed,
        skipped=acc.skipped,
        failures=tuple(acc.failures),
        cancelled=acc.cancelled,
    )


def _process_accepts_variant_index(process: Callable[..., np.ndarray]) -> bool:
    """True if `process` will accept a second positional ``variant_index``
    argument — lets callers pass a richer ``ProcessFunc`` without forcing
    every old single-arg caller to opt in."""
    try:
        import inspect

        sig = inspect.signature(process)
    except (TypeError, ValueError):
        # C functions / built-ins without a signature — assume the simple
        # single-arg shape.
        return False
    positional_kinds = (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.VAR_POSITIONAL,
    )
    positional = [p for p in sig.parameters.values() if p.kind in positional_kinds]
    if not positional:
        return False
    if positional[0].kind == inspect.Parameter.VAR_POSITIONAL:
        return True
    # Need at least two positional slots (image + variant_index).
    return len(positional) >= 2 or any(
        p.kind == inspect.Parameter.VAR_POSITIONAL for p in positional
    )


def make_request(
    *,
    input_dir: Path,
    output_dir: Path,
    process: ProcessFunc,
    output_suffix: str = "",
    overwrite: bool = False,
    paths: Iterable[Path] | None = None,
) -> BatchRequest:
    """Helper that auto-discovers images in `input_dir` unless an explicit
    `paths` iterable is given. Useful for both tests and the UI."""
    input_paths = tuple(paths) if paths is not None else discover_images(input_dir)
    return BatchRequest(
        input_paths=input_paths,
        output_dir=output_dir,
        process=process,
        output_suffix=output_suffix,
        overwrite=overwrite,
    )
