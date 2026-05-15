from __future__ import annotations

from pathlib import Path

import numpy as np

from cvstudio.ai import cache_storage
from cvstudio.ai.hf_owlvit import Detection
from cvstudio.operations import ai as ai_op


def _img(seed: int = 0) -> np.ndarray:
    """Reproducible test image — the bytes-level sha1 is the cache key,
    so we need a stable input rather than np.random."""
    return np.full((8, 8, 3), seed, dtype=np.uint8)


def test_round_trip_vlm_and_clip_caches(tmp_path: Path) -> None:
    ai_op.clear_cache()
    img = _img(10)

    vlm_key = ai_op._cache_key(img, "what is this?", "llava", 0.2)
    ai_op._cache_put(vlm_key, "a small grey square")

    clip_key = ai_op._clip_cache_key(img, ("cat", "dog"), "openai/clip")
    ai_op._clip_cache_put(clip_key, "cat 0.91, dog 0.09")

    cache_path = tmp_path / "ai_cache.json"
    cache_storage.save_caches(cache_path, ai_op.all_backends())
    assert cache_path.exists()

    ai_op.clear_cache()
    assert ai_op._cache_get(vlm_key) is None
    assert ai_op._clip_cache_get(clip_key) is None

    loaded = cache_storage.load_caches(cache_path, ai_op.all_backends())
    assert loaded == 2
    assert ai_op._cache_get(vlm_key) == "a small grey square"
    assert ai_op._clip_cache_get(clip_key) == "cat 0.91, dog 0.09"

    ai_op.clear_cache()


def test_round_trip_owlvit_detections(tmp_path: Path) -> None:
    ai_op.clear_cache()
    img = _img(20)

    detect_key = ai_op._detect_cache_key(
        img, ("a photo of a person",), "google/owlvit", 0.1
    )
    detections = [
        Detection(label="a photo of a person", score=0.91, box=(10, 20, 100, 200)),
        Detection(label="a photo of a person", score=0.55, box=(150, 50, 250, 200)),
    ]
    ai_op._detect_cache_put(detect_key, detections)

    cache_path = tmp_path / "ai_cache.json"
    cache_storage.save_caches(cache_path, ai_op.all_backends())

    ai_op.clear_cache()
    cache_storage.load_caches(cache_path, ai_op.all_backends())
    restored = ai_op._detect_cache_get(detect_key)
    assert isinstance(restored, list)
    assert restored == detections  # frozen dataclass equality

    ai_op.clear_cache()


def test_round_trip_owlvit_error_cached_as_string(tmp_path: Path) -> None:
    ai_op.clear_cache()
    img = _img(30)
    key = ai_op._detect_cache_key(img, ("x",), "m", 0.1)
    ai_op._detect_cache_put(key, "[Setup] install transformers + torch")

    cache_path = tmp_path / "ai_cache.json"
    cache_storage.save_caches(cache_path, ai_op.all_backends())

    ai_op.clear_cache()
    cache_storage.load_caches(cache_path, ai_op.all_backends())
    assert ai_op._detect_cache_get(key) == "[Setup] install transformers + torch"
    ai_op.clear_cache()


def test_load_missing_file_returns_zero(tmp_path: Path) -> None:
    count = cache_storage.load_caches(
        tmp_path / "nope.json", ai_op.all_backends()
    )
    assert count == 0


def test_load_malformed_json_does_not_crash(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("not-json-at-all{{", encoding="utf-8")
    count = cache_storage.load_caches(path, ai_op.all_backends())
    assert count == 0


def test_load_version_mismatch_ignored(tmp_path: Path) -> None:
    path = tmp_path / "old.json"
    path.write_text(
        '{"version": 999, "caches": {"vlm": [[["sha", "p", "m", 0.0], "x"]]}}',
        encoding="utf-8",
    )
    ai_op.clear_cache()
    count = cache_storage.load_caches(path, ai_op.all_backends())
    assert count == 0


def test_save_creates_parent_directory(tmp_path: Path) -> None:
    nested = tmp_path / "deeply" / "nested" / "dir" / "cache.json"
    ai_op.clear_cache()
    ai_op._cache_put(ai_op._cache_key(_img(1), "p", "m", 0.0), "ok")
    cache_storage.save_caches(nested, ai_op.all_backends())
    assert nested.exists()
    ai_op.clear_cache()


def test_default_cache_path_returns_writable_location() -> None:
    """Path resolution must always yield SOMETHING — it doesn't have to
    exist yet, but `path.parent.mkdir(...)` must be able to create it."""
    p = cache_storage.default_cache_path()
    assert isinstance(p, Path)
    assert p.name == "ai_cache.json"
