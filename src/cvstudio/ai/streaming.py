"""Streaming state + cross-thread re-render bus + cancellation.

Background workers stream tokens from an LLM/VLM backend into the
module-level partial store. Each token append fires the `bus().progress`
Qt signal so the GUI thread can re-run the pipeline and the banner shows
the latest text. Because the pipeline contract is synchronous
(`image -> image`), the actual op (`_vlm_query`) re-reads the partial
store on each pipeline run instead of waiting for the stream to finish.

Cancellation model:
    * Each active stream has a `threading.Event` flag. Workers poll it
      between tokens and bail out without writing to cache when set.
    * Streams are associated with the pipeline node that spawned them
      (via the `set_current_node` thread-local set by the pipeline
      worker before each `func()` call). When a node spawns a new
      stream, it can cancel its own older streams without disturbing
      streams owned by other nodes.

Thread safety:
    * `_partials`, `_active`, `_cancels`, `_node_streams` are guarded by
      a single RLock so observers see consistent snapshots.
    * The Qt signal is emitted from worker threads — PySide6 routes it
      to GUI-thread slots via auto-queued connections.
"""

from __future__ import annotations

import threading
from typing import Hashable

from PySide6.QtCore import QObject, Signal


class StreamingBus(QObject):
    """Single signal — `progress` — emitted whenever a streaming worker
    has new partial text. Multiple emissions per frame are fine: the GUI
    side debounces them through its existing 120 ms preview debouncer."""

    progress = Signal()


_bus = StreamingBus()
_lock = threading.RLock()
_partials: dict[Hashable, str] = {}
_active: set[Hashable] = set()
_cancels: dict[Hashable, threading.Event] = {}
_node_streams: dict[str, set[Hashable]] = {}
_node_display: dict[str, str] = {}
_node_local = threading.local()


def bus() -> StreamingBus:
    return _bus


# --------------------------------------------------------------- node context


def set_current_node(node_id: str | None) -> None:
    """Pipeline worker calls this before each op's `func()` so the op can
    discover which pipeline node it represents and scope its streams
    accordingly. Pass `None` after the call to leave no stale context."""
    _node_local.value = node_id


def current_node() -> str | None:
    return getattr(_node_local, "value", None)


# --------------------------------------------------------------- partial store


def get_partial(key: Hashable) -> str | None:
    with _lock:
        return _partials.get(key)


def is_streaming(key: Hashable) -> bool:
    with _lock:
        return key in _active


def begin_stream(key: Hashable, *, node_id: str | None = None) -> bool:
    """Reserve `key` for a new streaming worker. Returns False if another
    worker is already streaming this key. When `node_id` is given the
    stream is tagged with it so `cancel_node_streams_except` can find
    siblings to cancel later."""
    with _lock:
        if key in _active:
            return False
        _active.add(key)
        _partials[key] = ""
        _cancels[key] = threading.Event()
        if node_id is not None:
            _node_streams.setdefault(node_id, set()).add(key)
    return True


def append_token(key: Hashable, token: str) -> None:
    """Append a streamed token to the partial text for `key`. No-op when
    the stream has already been ended or cancelled."""
    with _lock:
        if key not in _active:
            return
        event = _cancels.get(key)
        if event is not None and event.is_set():
            return
        _partials[key] = _partials.get(key, "") + token
    _bus.progress.emit()


def end_stream(key: Hashable) -> str:
    """Finalize a streaming worker — clear all bookkeeping (active flag,
    cancel event, node mapping) and return the accumulated text. The
    caller decides whether to commit the final text to a longer-lived
    cache."""
    with _lock:
        _active.discard(key)
        text = _partials.pop(key, "")
        _cancels.pop(key, None)
        for keys in _node_streams.values():
            keys.discard(key)
    _bus.progress.emit()
    return text


# --------------------------------------------------------------- cancellation


def cancel_stream(key: Hashable) -> bool:
    """Request the worker for `key` to stop emitting tokens at its next
    checkpoint. Returns True if a cancellable stream existed."""
    with _lock:
        event = _cancels.get(key)
    if event is None:
        return False
    event.set()
    return True


def cancel_node_streams_except(node_id: str, current_key: Hashable | None) -> int:
    """Cancel every stream tagged with `node_id` whose key differs from
    `current_key`. Returns the number of streams cancelled. Used by
    `_vlm_query` to invalidate in-flight work after the user changes
    prompt/model/temperature on a node."""
    with _lock:
        keys = set(_node_streams.get(node_id, ()))
        targets = {k for k in keys if k != current_key}
        events = [_cancels[k] for k in targets if k in _cancels]
    for event in events:
        event.set()
    return len(events)


def is_cancelled(key: Hashable) -> bool:
    with _lock:
        event = _cancels.get(key)
    return event is not None and event.is_set()


def reset() -> None:
    """Test hook — wipe streaming state without touching downstream
    caches. Not exposed in the UI."""
    with _lock:
        _partials.clear()
        _active.clear()
        _cancels.clear()
        _node_streams.clear()
        _node_display.clear()
    set_current_node(None)


# --------------------------------------------------------------- node display
#
# AI ops write their user-facing response text here keyed by pipeline
# node id; the ParameterPanel's "AI Response" area reads it for the
# currently-selected node and re-renders on every `progress` emission.
# This is the seam that lets us keep text out of the image — the op's
# pipeline return value can stay an unmodified ndarray.


def set_node_display(node_id: str | None, text: str) -> None:
    """Publish the response text for `node_id` and notify the GUI.
    A None `node_id` is treated as a no-op so ops called directly from
    tests (without a pipeline worker setting the thread-local) don't
    crash."""
    if node_id is None:
        return
    with _lock:
        _node_display[node_id] = text
    _bus.progress.emit()


def get_node_display(node_id: str | None) -> str | None:
    if node_id is None:
        return None
    with _lock:
        return _node_display.get(node_id)


def clear_node_display(node_id: str | None) -> None:
    """Drop the response text for `node_id` — e.g. when the node is
    removed from the pipeline. Bus is not signalled (the panel will
    refresh on the next selection change)."""
    if node_id is None:
        return
    with _lock:
        _node_display.pop(node_id, None)
