"""On-disk persistence for AI op caches.

A single JSON file holds every backend's cache so model responses (the
expensive part: an Ollama VLM call can take 10+ seconds, a CLIP run
loads a 600MB checkpoint) survive app restarts. The format is
intentionally readable so users can spot-check or hand-edit it:

    {
        "version": 1,
        "caches": {
            "vlm":    [[["sha1", "prompt", "model", 0.2], "the reply"], ...],
            "clip":   [[["sha1", ["label-a", "label-b"], "model"], "label-a 0.91, ..."], ...],
            "owlvit": [[["sha1", ["prompt"], "model", 0.1],
                        {"kind": "detections", "items": [...]}], ...],
            "blip2":  [[["sha1", "model", 30], "a brown dog"], ...]
        }
    }

Tuples are flattened to lists at write time (`backend._tuple_to_list`)
and rebuilt at read time so cache keys stay hashable. Corrupt or
schema-mismatched files are silently treated as "no cache" rather than
crashing the app at launch — a busted cache is recoverable; failing to
start the editor is not.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cvsandbox.ai.backend import AIBackend

logger = logging.getLogger(__name__)

CACHE_VERSION = 1


def default_cache_path() -> Path:
    """Resolve the AI cache path via Qt's per-OS app data location.
    Falls back to ``~/.cvsandbox/ai_cache.json`` when Qt is missing
    (e.g. headless CI)."""
    try:
        from PySide6.QtCore import QStandardPaths

        base = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.AppDataLocation
        )
        if base:
            return Path(base) / "ai_cache.json"
    except ImportError:
        pass
    return Path.home() / ".cvsandbox" / "ai_cache.json"


def save_caches(path: Path, backends: "dict[str, AIBackend]") -> None:
    """Snapshot every registered backend's cache to `path`. Atomic
    write: dump to a sibling tempfile and `replace` so a crashed write
    cannot corrupt the existing on-disk cache."""
    payload = {
        "version": CACHE_VERSION,
        "caches": {
            name: backend.cache_items_serializable()
            for name, backend in backends.items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a tempfile in the same directory so `replace` is atomic
    # on every OS (cross-fs rename would fall back to copy + delete).
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        delete=False,
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".tmp",
    ) as tmp:
        json.dump(payload, tmp, indent=2)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def load_caches(path: Path, backends: "dict[str, AIBackend]") -> int:
    """Restore each backend's cache from `path`. Returns the total
    number of entries loaded across all backends. Missing file,
    malformed JSON, or a version mismatch all return 0 silently —
    those are recoverable, the user will just re-run their queries."""
    if not path.exists():
        return 0
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not load AI cache from %s: %s", path, exc)
        return 0
    if not isinstance(data, dict) or data.get("version") != CACHE_VERSION:
        logger.warning(
            "AI cache at %s has unsupported version %r; ignoring.",
            path,
            data.get("version") if isinstance(data, dict) else None,
        )
        return 0

    caches = data.get("caches")
    if not isinstance(caches, dict):
        return 0

    total = 0
    for name, backend in backends.items():
        entries = caches.get(name, [])
        if not isinstance(entries, list):
            continue
        backend.cache_load_serialized(entries)
        total += len(backend.cache_items())
    return total
