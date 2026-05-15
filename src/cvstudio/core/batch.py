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

ProcessFunc = Callable[[np.ndarray], np.ndarray]
ProgressCallback = Callable[[int, int, Path], None]
CancelCheck = Callable[[], bool]


@dataclass(frozen=True)
class BatchRequest:
    input_paths: tuple[Path, ...]
    output_dir: Path
    process: ProcessFunc
    output_suffix: str = ""
    overwrite: bool = False

    def output_path_for(self, input_path: Path) -> Path:
        return self.output_dir / f"{input_path.stem}{self.output_suffix}{input_path.suffix}"


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


def execute_batch(
    request: BatchRequest,
    *,
    on_progress: ProgressCallback | None = None,
    cancel_check: CancelCheck | None = None,
) -> BatchResult:
    request.output_dir.mkdir(parents=True, exist_ok=True)
    total = len(request.input_paths)
    acc = _Accumulator()

    for index, input_path in enumerate(request.input_paths):
        if cancel_check is not None and cancel_check():
            acc.cancelled = True
            break

        if on_progress is not None:
            on_progress(index, total, input_path)

        output_path = request.output_path_for(input_path)
        if output_path.exists() and not request.overwrite:
            acc.skipped += 1
            continue

        try:
            image = read_image(input_path)
            if image is None:
                raise OSError(f"unreadable image: {input_path}")
            result = request.process(image)
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

    if on_progress is not None and not acc.cancelled:
        # Final tick — useful for the UI to set "100% done".
        on_progress(total, total, request.input_paths[-1] if request.input_paths else Path())

    return BatchResult(
        completed=acc.completed,
        skipped=acc.skipped,
        failures=tuple(acc.failures),
        cancelled=acc.cancelled,
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
