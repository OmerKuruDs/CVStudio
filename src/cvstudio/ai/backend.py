"""Shared scaffolding for AI pipeline ops.

Every AI op (VLM Q&A, CLIP classify, OWL-ViT detect, BLIP-2 caption)
needs the same machinery:

  * an LRU cache keyed on `(image_hash, *op-specific-params)`,
  * a per-node "Run authorization" gate so expensive backends don't
    fire on every keystroke,
  * a daemon worker that runs the inference, writes to the cache, and
    publishes status text to the per-node display store,
  * auto-cancellation of older streams on the same node when params
    change.

`AIBackend` captures that flow. Concrete backends supply only what
varies: `validate(params)`, `make_key(image, params)`,
`run(key, image, params, node_id)`, `format_display(result)`, and
(optionally) `render(image, result, params)` for ops that draw pixels
on the canvas — OWL-ViT being the only current case.

`StreamingAIBackend` is a thin variant for backends that yield tokens
incrementally (Ollama VLM): subclasses implement `iter_tokens` instead
of `run`, and the base loop updates the partial display after each
token so the user sees the reply forming in real time.
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable, Iterator
from typing import Any, Generic, TypeVar

import numpy as np

from cvstudio.ai import streaming
from cvstudio.ai.hf_clip import HFExtrasMissing

logger = logging.getLogger(__name__)

CACHE_MAX = 64
PENDING_HINT = "Press Run in the parameter panel to query the model."
THINKING_PLACEHOLDER = "Thinking…"
STREAM_CURSOR = "▌"

ResultT = TypeVar("ResultT")
KeyT = TypeVar("KeyT", bound=tuple)


# --------------------------------------------------------------- authorization


_AUTHORIZED_NODES: set[str] = set()
_AUTH_LOCK = threading.RLock()


def authorize_node(node_id: str) -> None:
    """Grant `node_id` a one-shot permission to spawn a new backend
    call on its next pipeline run. Consumed by `_consume_auth`. Called
    by the ParameterPanel's Run button via MainWindow."""
    with _AUTH_LOCK:
        _AUTHORIZED_NODES.add(node_id)


def consume_auth(node_id: str | None, *, force: bool = False) -> bool:
    """Return True (and clear the one-shot flag) if `node_id` is
    currently authorized. `force=True` bypasses the check entirely —
    used by the `auto_run` parameter so video pipelines can run a model
    every frame without a Run click between each.

    `node_id is None` (direct test invocation outside the pipeline
    worker) is also treated as authorized so tests can call op funcs
    without setting up the thread-local context."""
    if force or node_id is None:
        return True
    with _AUTH_LOCK:
        if node_id in _AUTHORIZED_NODES:
            _AUTHORIZED_NODES.discard(node_id)
            return True
    return False


def clear_authorizations() -> None:
    """Test hook — wipe pending auth flags. Not exposed in the UI."""
    with _AUTH_LOCK:
        _AUTHORIZED_NODES.clear()


# --------------------------------------------------------------- threading


def run_in_thread(target: Callable[[], None]) -> None:
    """Default worker spawner — daemon thread so the pipeline worker
    never blocks on a multi-second model call. Tests swap this for a
    synchronous runner via monkeypatch."""
    threading.Thread(target=target, daemon=True).start()


# --------------------------------------------------------------- backend base


class AIBackend(ABC, Generic[KeyT, ResultT]):
    """Common scaffolding for an AI op. Subclasses override the four
    `validate`, `make_key`, `run`, `format_display` hooks (plus the
    optional `render`) and call `execute()` from the pipeline spec."""

    running_placeholder: str = THINKING_PLACEHOLDER

    def __init__(self) -> None:
        self._cache: "OrderedDict[KeyT, ResultT | str]" = OrderedDict()
        self._cache_lock = threading.RLock()
        # Tests swap this for a synchronous runner; callers that
        # configure their own dispatch (e.g. an injected thread pool)
        # can do the same via `set_runner`.
        self._runner: Callable[[Callable[[], None]], None] = run_in_thread

    def set_runner(
        self, runner: Callable[[Callable[[], None]], None]
    ) -> None:
        self._runner = runner

    # ---- subclass hooks ------------------------------------------------

    @abstractmethod
    def validate(self, params: dict[str, Any]) -> str | None:
        """Return a user-visible hint when `params` are invalid; the op
        publishes it to the AI Response panel and skips inference.
        Return None when params are good to go."""

    @abstractmethod
    def make_key(self, image: np.ndarray, params: dict[str, Any]) -> KeyT:
        """Build the cache key for this image+params combination."""

    @abstractmethod
    def run(
        self,
        key: KeyT,
        image: np.ndarray,
        params: dict[str, Any],
        node_id: str | None,
    ) -> ResultT | None:
        """Execute the model. Called from a worker thread. Returns the
        result on success or `None` when cancelled mid-flight. Errors
        bubble up — `_spawn_worker` converts them into cached error
        messages so a failing call doesn't retry-storm."""

    @abstractmethod
    def format_display(self, result: ResultT) -> str:
        """Convert a successful result to text for the side panel."""

    def render(
        self,
        image: np.ndarray,
        result: ResultT,
        params: dict[str, Any],
    ) -> np.ndarray:
        """Default = passthrough. Override for ops that draw pixels
        (OWL-ViT bounding boxes)."""
        del result, params
        return image

    def format_error(self, exc: Exception) -> str | None:
        """Subclass-supplied error message for a domain-specific
        exception. Return None to fall through to the generic
        `[Error] ClassName: msg` formatting."""
        del exc
        return None

    # ---- cache ---------------------------------------------------------

    def cache_get(self, key: KeyT) -> "ResultT | str | None":
        with self._cache_lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        return None

    def cache_put(self, key: KeyT, value: "ResultT | str") -> None:
        with self._cache_lock:
            self._cache[key] = value
            self._cache.move_to_end(key)
            while len(self._cache) > CACHE_MAX:
                self._cache.popitem(last=False)

    def cache_clear(self) -> None:
        with self._cache_lock:
            self._cache.clear()

    def cache_items(self) -> "list[tuple[KeyT, ResultT | str]]":
        with self._cache_lock:
            return list(self._cache.items())

    # ---- persistence ---------------------------------------------------

    def serialize_value(self, value: "ResultT | str") -> Any:
        """Default: round-trip strings as-is. Subclasses that store
        structured results override this and `deserialize_value`."""
        return value

    def deserialize_value(self, raw: Any) -> "ResultT | str | None":
        """Default: accept any string. Return None to skip a corrupt
        entry rather than crashing the load."""
        if isinstance(raw, str):
            return raw
        return None

    def cache_items_serializable(self) -> list[list[Any]]:
        """Encode every cache entry as a `[key_as_list, value]` pair
        for JSON persistence. Tuples are flattened to nested lists."""
        return [[_tuple_to_list(k), self.serialize_value(v)] for k, v in self.cache_items()]

    def cache_load_serialized(self, entries: list[Any]) -> None:
        """Replace the in-memory cache with `entries` (as produced by
        `cache_items_serializable`). Malformed entries are dropped
        with a log warning — a broken on-disk cache should not crash
        the app at launch."""
        loaded: list[tuple[KeyT, "ResultT | str"]] = []
        for entry in entries:
            if not isinstance(entry, list) or len(entry) != 2:
                continue
            raw_key, raw_value = entry
            key = _list_to_tuple(raw_key)
            value = self.deserialize_value(raw_value)
            if value is None:
                continue
            loaded.append((key, value))  # type: ignore[arg-type]
        with self._cache_lock:
            self._cache.clear()
            # Honour the LRU cap even when the persisted file is bigger.
            for k, v in loaded[-CACHE_MAX:]:
                self._cache[k] = v

    # ---- spawn ---------------------------------------------------------

    def _spawn_worker(
        self,
        key: KeyT,
        image: np.ndarray,
        params: dict[str, Any],
        node_id: str | None,
    ) -> None:
        def _worker() -> None:
            try:
                if streaming.is_cancelled(key):
                    streaming.end_stream(key)
                    return
                result = self.run(key, image, params, node_id)
                if result is None or streaming.is_cancelled(key):
                    streaming.end_stream(key)
                    return
                streaming.end_stream(key)
                self.cache_put(key, result)
                streaming.set_node_display(node_id, self.format_display(result))
            except Exception as exc:  # noqa: BLE001 — converted to user message below
                message = self._exception_to_message(exc)
                cancelled = streaming.is_cancelled(key)
                streaming.end_stream(key)
                if not cancelled:
                    self.cache_put(key, message)
                    streaming.set_node_display(node_id, message)

        self._runner(_worker)

    def _exception_to_message(self, exc: Exception) -> str:
        if isinstance(exc, HFExtrasMissing):
            return f"[Setup] {exc}"
        custom = self.format_error(exc)
        if custom is not None:
            return custom
        return f"[Error] {type(exc).__name__}: {exc}"

    # ---- main entry point ---------------------------------------------

    def execute(self, image: np.ndarray, **params: Any) -> np.ndarray:
        node_id = streaming.current_node()

        hint = self.validate(params)
        if hint is not None:
            streaming.set_node_display(node_id, hint)
            return image

        key = self.make_key(image, params)

        cached = self.cache_get(key)
        if cached is not None:
            if node_id is not None:
                streaming.cancel_node_streams_except(node_id, key)
            return self._render_cached(image, cached, params, node_id)

        if streaming.is_streaming(key):
            partial = streaming.get_partial(key) or self.running_placeholder
            streaming.set_node_display(node_id, partial)
            return image

        # auto_run bypasses the manual Run-button gate. Used by video
        # frame-by-frame pipelines where the user can't click for every
        # frame. Auto-cancel still trims a backlog if the model can't
        # keep up with the source's framerate.
        auto = bool(params.get("auto_run", False))
        if not consume_auth(node_id, force=auto):
            streaming.set_node_display(node_id, PENDING_HINT)
            return image

        if node_id is not None:
            streaming.cancel_node_streams_except(node_id, key)

        if streaming.begin_stream(key, node_id=node_id):
            self._spawn_worker(key, image.copy(), dict(params), node_id)

        # Re-check the cache after spawning — in sync test mode the
        # worker has already populated it.
        cached = self.cache_get(key)
        if cached is not None:
            return self._render_cached(image, cached, params, node_id)
        partial = streaming.get_partial(key) or self.running_placeholder
        streaming.set_node_display(node_id, partial)
        return image

    def _render_cached(
        self,
        image: np.ndarray,
        cached: "ResultT | str",
        params: dict[str, Any],
        node_id: str | None,
    ) -> np.ndarray:
        if isinstance(cached, str):
            # Error path OR a backend whose ResultT is also str. Both
            # render the same way — text-only.
            streaming.set_node_display(node_id, cached)
            return image
        streaming.set_node_display(node_id, self.format_display(cached))
        return self.render(image, cached, params)


# --------------------------------------------------------------- streaming variant


class StreamingAIBackend(AIBackend[KeyT, str]):
    """For backends that yield tokens incrementally (Ollama VLM). The
    base loop pulls each token, updates the per-key partial, and
    publishes a live-updating snapshot (with cursor) to the side
    panel. Returns the accumulated text on completion."""

    running_placeholder = THINKING_PLACEHOLDER

    @abstractmethod
    def iter_tokens(
        self, image: np.ndarray, params: dict[str, Any]
    ) -> Iterator[str]:
        """Yield response tokens one by one. Connection errors raise
        before the first yield; mid-stream backend errors raise from
        inside the iterator and are caught by the worker."""

    def run(
        self,
        key: KeyT,
        image: np.ndarray,
        params: dict[str, Any],
        node_id: str | None,
    ) -> str | None:
        for token in self.iter_tokens(image, params):
            if streaming.is_cancelled(key):
                return None
            streaming.append_token(key, token)
            partial = streaming.get_partial(key) or ""
            streaming.set_node_display(node_id, partial + STREAM_CURSOR)
        if streaming.is_cancelled(key):
            return None
        final = (streaming.get_partial(key) or "").strip()
        return final if final else "(empty reply)"

    def format_display(self, result: str) -> str:
        return result


# --------------------------------------------------------------- helpers


def _tuple_to_list(obj: Any) -> Any:
    """Recursively turn tuples into lists so the result is JSON-safe."""
    if isinstance(obj, tuple):
        return [_tuple_to_list(x) for x in obj]
    return obj


def _list_to_tuple(obj: Any) -> Any:
    """Inverse of `_tuple_to_list` — used at load time so cache keys
    are hashable again."""
    if isinstance(obj, list):
        return tuple(_list_to_tuple(x) for x in obj)
    return obj
