from __future__ import annotations

import io
import json
from collections.abc import Iterator

import numpy as np
import pytest

from cvsandbox.ai import ollama_client, streaming
from cvsandbox.operations import ai as ai_op
from cvsandbox.operations.ai import VLM_QUERY


@pytest.fixture(autouse=True)
def _reset_state() -> None:
    ai_op.clear_cache()
    streaming.reset()


@pytest.fixture
def sync_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the streaming worker synchronously so cache state is observable
    right after `_vlm_query` returns."""
    monkeypatch.setattr(ai_op, "_run_in_thread", lambda fn: fn())


# --------------------------------------------------------------- ollama client


class _FakeOpener:
    """Captures the request body and returns a canned JSON payload."""

    def __init__(self, response_text: str = "a cat sitting on a mat") -> None:
        self.response_text = response_text
        self.last_url: str | None = None
        self.last_body: dict | None = None

    def open(self, request, timeout):  # noqa: ARG002
        self.last_url = request.full_url
        self.last_body = json.loads(request.data.decode("utf-8"))
        payload = {"model": self.last_body["model"], "response": self.response_text}
        return io.BytesIO(json.dumps(payload).encode("utf-8"))


class _FakeStreamingOpener:
    """Returns a sequence of NDJSON lines that look like Ollama's stream=true
    response. Each line is independently json-decoded by `stream_generate`."""

    def __init__(self, lines: list[bytes]) -> None:
        self.lines = list(lines)
        self.last_body: dict | None = None

    def open(self, request, timeout):  # noqa: ARG002
        self.last_body = json.loads(request.data.decode("utf-8"))
        # `iter(BytesIO)` yields one line per `\n` — same shape as urlopen().
        joined = b"".join(self.lines)
        return io.BytesIO(joined)


def test_ollama_generate_posts_image_and_prompt() -> None:
    fake = _FakeOpener("a red square")
    img = np.full((8, 8, 3), 200, dtype=np.uint8)
    response = ollama_client.generate(
        "What is this?",
        img,
        opener=fake,  # type: ignore[arg-type]
    )
    assert response.text == "a red square"
    assert fake.last_body is not None
    assert fake.last_body["stream"] is False
    assert isinstance(fake.last_body["images"][0], str)


def test_ollama_generate_raises_on_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    import urllib.error

    def _boom(*_args, **_kwargs):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(ollama_client.urllib.request, "urlopen", _boom)
    with pytest.raises(ollama_client.OllamaError, match="Could not reach"):
        ollama_client.generate("hi", np.zeros((4, 4, 3), dtype=np.uint8))


def test_stream_generate_yields_each_token() -> None:
    fake = _FakeStreamingOpener(
        [
            b'{"response": "Hello", "done": false}\n',
            b'{"response": ", ", "done": false}\n',
            b'{"response": "world", "done": false}\n',
            b'{"response": "!", "done": true}\n',
        ]
    )
    img = np.full((8, 8, 3), 100, dtype=np.uint8)
    tokens = list(
        ollama_client.stream_generate("hi", img, opener=fake)  # type: ignore[arg-type]
    )
    assert tokens == ["Hello", ", ", "world", "!"]
    assert fake.last_body is not None
    assert fake.last_body["stream"] is True


def test_stream_generate_raises_on_mid_stream_error() -> None:
    fake = _FakeStreamingOpener(
        [
            b'{"response": "Loading", "done": false}\n',
            b'{"error": "model not found"}\n',
        ]
    )
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    iterator = ollama_client.stream_generate("hi", img, opener=fake)  # type: ignore[arg-type]
    assert next(iterator) == "Loading"
    with pytest.raises(ollama_client.OllamaError, match="model not found"):
        next(iterator)


def test_stream_generate_skips_malformed_lines() -> None:
    fake = _FakeStreamingOpener(
        [
            b"\n",  # empty
            b"not-json\n",  # malformed — should be ignored
            b'{"response": "ok", "done": true}\n',
        ]
    )
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    tokens = list(
        ollama_client.stream_generate("hi", img, opener=fake)  # type: ignore[arg-type]
    )
    assert tokens == ["ok"]


# --------------------------------------------------------------- streaming state


def test_streaming_begin_blocks_duplicate_for_same_key() -> None:
    assert streaming.begin_stream("k1") is True
    assert streaming.begin_stream("k1") is False
    assert streaming.is_streaming("k1") is True
    streaming.end_stream("k1")
    assert streaming.is_streaming("k1") is False


def test_streaming_append_and_finalize_accumulates_text() -> None:
    streaming.begin_stream("k2")
    streaming.append_token("k2", "Hello")
    streaming.append_token("k2", " world")
    assert streaming.get_partial("k2") == "Hello world"
    final = streaming.end_stream("k2")
    assert final == "Hello world"
    assert streaming.get_partial("k2") is None


def test_streaming_append_after_end_is_dropped() -> None:
    streaming.begin_stream("k3")
    streaming.end_stream("k3")
    streaming.append_token("k3", "late")  # race-safe drop
    assert streaming.get_partial("k3") is None


# --------------------------------------------------------------- cancellation


def test_cancel_stream_marks_event() -> None:
    streaming.begin_stream("ck1")
    assert streaming.is_cancelled("ck1") is False
    assert streaming.cancel_stream("ck1") is True
    assert streaming.is_cancelled("ck1") is True
    streaming.end_stream("ck1")
    # After end, the cancel event is cleared — nothing to cancel for a
    # gone key.
    assert streaming.cancel_stream("ck1") is False


def test_cancel_stream_drops_subsequent_tokens() -> None:
    streaming.begin_stream("ck2")
    streaming.append_token("ck2", "before")
    streaming.cancel_stream("ck2")
    streaming.append_token("ck2", "after")  # silently dropped
    assert streaming.get_partial("ck2") == "before"


def test_cancel_node_streams_spares_current_key() -> None:
    streaming.begin_stream("old_key", node_id="N1")
    streaming.begin_stream("new_key", node_id="N1")
    streaming.begin_stream("other_node", node_id="N2")

    n = streaming.cancel_node_streams_except("N1", current_key="new_key")
    assert n == 1
    assert streaming.is_cancelled("old_key") is True
    assert streaming.is_cancelled("new_key") is False
    assert streaming.is_cancelled("other_node") is False
    streaming.end_stream("old_key")
    streaming.end_stream("new_key")
    streaming.end_stream("other_node")


def test_current_node_thread_local_set_and_clear() -> None:
    assert streaming.current_node() is None
    streaming.set_current_node("nA")
    assert streaming.current_node() == "nA"
    streaming.set_current_node(None)
    assert streaming.current_node() is None


def test_vlm_query_cancels_previous_stream_on_prompt_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """User changes the prompt before the first reply lands. The op must
    cancel the in-flight worker for the old prompt and start a new one
    for the new prompt — without the old worker poisoning the cache."""
    img = np.full((180, 360, 3), 200, dtype=np.uint8)
    spawned: list[tuple] = []

    def _record_spawn(_fn):
        spawned.append(("spawned",))

    monkeypatch.setattr(ai_op, "_run_in_thread", _record_spawn)
    streaming.set_current_node("nVLM")
    try:
        ai_op.authorize_node("nVLM")
        _run_op(img, prompt="What is this?")
        old_key = ai_op._cache_key(img, "What is this?", "llava", 0.2)
        assert streaming.is_streaming(old_key)
        assert streaming.is_cancelled(old_key) is False

        ai_op.authorize_node("nVLM")
        _run_op(img, prompt="Count the objects.")
        new_key = ai_op._cache_key(img, "Count the objects.", "llava", 0.2)
        assert streaming.is_cancelled(old_key) is True
        assert streaming.is_streaming(new_key)
        assert streaming.is_cancelled(new_key) is False
    finally:
        streaming.set_current_node(None)


def test_vlm_query_does_not_cancel_other_node_streams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two VLM nodes streaming on the same image must coexist — switching
    pipeline focus from one to the other does not cancel the other's
    in-flight work."""
    img = np.full((180, 360, 3), 200, dtype=np.uint8)
    monkeypatch.setattr(ai_op, "_run_in_thread", lambda _fn: None)

    streaming.set_current_node("nodeA")
    try:
        ai_op.authorize_node("nodeA")
        _run_op(img, prompt="prompt-a")
    finally:
        streaming.set_current_node(None)
    key_a = ai_op._cache_key(img, "prompt-a", "llava", 0.2)
    assert streaming.is_streaming(key_a)

    streaming.set_current_node("nodeB")
    try:
        ai_op.authorize_node("nodeB")
        _run_op(img, prompt="prompt-b")
    finally:
        streaming.set_current_node(None)
    key_b = ai_op._cache_key(img, "prompt-b", "llava", 0.2)
    # Both streams active; neither cancelled.
    assert streaming.is_streaming(key_a)
    assert streaming.is_cancelled(key_a) is False
    assert streaming.is_streaming(key_b)
    assert streaming.is_cancelled(key_b) is False


def test_cancelled_stream_does_not_poison_cache(
    sync_threads, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worker that observes its cancel flag mid-iteration must exit
    without writing to the cache so subsequent renders can retry."""
    img = np.full((180, 360, 3), 200, dtype=np.uint8)
    key = ai_op._cache_key(img, "What is this?", "llava", 0.2)

    def _slow_stream(prompt, image, **_kwargs) -> Iterator[str]:  # noqa: ARG001
        yield "first token"
        # Cancel between tokens — simulates the user changing prompts
        # while the worker was producing.
        streaming.cancel_stream(key)
        yield "second token (should be dropped)"

    monkeypatch.setattr(ai_op, "stream_generate", _slow_stream)
    streaming.set_current_node("nVLM")
    try:
        _run_op(img)
    finally:
        streaming.set_current_node(None)

    assert ai_op._cache_get(key) is None  # cancelled streams don't cache
    assert streaming.is_streaming(key) is False


def test_vlm_query_renders_pending_hint_without_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inside a pipeline (current_node set) the op must NOT spawn until
    `authorize_node` has been called — typing in the prompt field cannot
    trigger Ollama queries on its own."""

    def _should_not_spawn(_fn):
        raise AssertionError("must not spawn without explicit authorization")

    monkeypatch.setattr(ai_op, "_run_in_thread", _should_not_spawn)
    img = np.full((180, 360, 3), 200, dtype=np.uint8)
    streaming.set_current_node("nVLM")
    try:
        out = _run_op(img)
    finally:
        streaming.set_current_node(None)
    assert out.shape == img.shape
    key = ai_op._cache_key(img, "What is this?", "llava", 0.2)
    assert streaming.is_streaming(key) is False
    assert ai_op._cache_get(key) is None  # nothing was generated


def test_vlm_query_consumes_auth_once_then_returns_to_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Authorization is one-shot: after the spawn, a subsequent render
    with a different uncached key must fall back to the pending-hint
    placeholder instead of spawning again."""
    spawned = 0

    def _count_spawn(_fn):
        nonlocal spawned
        spawned += 1

    monkeypatch.setattr(ai_op, "_run_in_thread", _count_spawn)
    img = np.full((80, 200, 3), 120, dtype=np.uint8)
    streaming.set_current_node("nOnce")
    try:
        ai_op.authorize_node("nOnce")
        _run_op(img, prompt="prompt-1")
        # Different prompt → different key. Without a fresh authorize,
        # the op must not spawn.
        _run_op(img, prompt="prompt-2")
    finally:
        streaming.set_current_node(None)
    assert spawned == 1


def test_vlm_query_cache_hit_does_not_require_auth() -> None:
    """A cached entry should be served even without authorization — once
    the user has paid for a query its result is theirs to revisit. The
    image is returned unchanged; the reply lives in the side panel via
    `streaming.get_node_display`."""
    img = np.full((60, 60, 3), 80, dtype=np.uint8)
    key = ai_op._cache_key(img, "What is this?", "llava", 0.2)
    ai_op._cache_put(key, "cached answer from previous session")

    streaming.set_current_node("nCached")
    try:
        out = _run_op(img)
    finally:
        streaming.set_current_node(None)
    assert np.array_equal(out, img)  # passthrough — no banner on the image
    assert (
        streaming.get_node_display("nCached")
        == "cached answer from previous session"
    )


# --------------------------------------------------------------- CLIP op


def _run_clip(img: np.ndarray, **overrides) -> np.ndarray:
    from cvsandbox.operations.ai import CLIP_CLASSIFY

    params = {
        "labels": "a photo of a cat, a photo of a dog, a photo of a car",
        "model_name": "openai/clip-vit-base-patch32",
        "top_k": 3,
        "font_scale": 0.6,
    }
    params.update(overrides)
    return CLIP_CLASSIFY.func(img, **params)


def test_parse_labels_splits_comma_and_semicolon_and_lines() -> None:
    parsed = ai_op._parse_labels("a, b ; c\nd\n  , e")
    assert parsed == ["a", "b", "c", "d", "e"]


def test_parse_labels_drops_empty() -> None:
    assert ai_op._parse_labels(" , , ") == []
    assert ai_op._parse_labels("") == []


def test_format_clip_result_respects_top_k() -> None:
    pairs = [("cat", 0.7), ("dog", 0.2), ("car", 0.1)]
    line = ai_op._format_clip_result(pairs, top_k=2)
    assert line == "cat 0.70, dog 0.20"


def test_clip_classify_empty_labels_shows_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    def _should_not_classify(*_args, **_kwargs):
        raise AssertionError("must not call hf_clip.classify without labels")

    from cvsandbox.ai import hf_clip

    monkeypatch.setattr(hf_clip, "classify", _should_not_classify)
    img = np.full((80, 240, 3), 100, dtype=np.uint8)
    out = _run_clip(img, labels="")
    assert out.shape == img.shape


def test_clip_classify_caches_result(
    sync_threads, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []

    def _fake_classify(image, labels, *, model_name):  # noqa: ARG001
        calls.append(list(labels))
        return [(label, 1.0 / len(labels)) for label in labels]

    from cvsandbox.ai import hf_clip

    monkeypatch.setattr(hf_clip, "classify", _fake_classify)
    img = np.full((120, 240, 3), 150, dtype=np.uint8)
    _run_clip(img)
    _run_clip(img)
    assert len(calls) == 1
    assert calls[0] == [
        "a photo of a cat",
        "a photo of a dog",
        "a photo of a car",
    ]


def test_clip_classify_requires_auth_in_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _should_not_spawn(_fn):
        raise AssertionError("CLIP must not spawn without authorization")

    monkeypatch.setattr(ai_op, "_run_in_thread", _should_not_spawn)
    img = np.full((180, 360, 3), 200, dtype=np.uint8)
    streaming.set_current_node("nClip")
    try:
        out = _run_clip(img)
    finally:
        streaming.set_current_node(None)
    assert out.shape == img.shape


def test_clip_classify_renders_top_labels_after_sync_run(
    sync_threads, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _fake_classify(image, labels, *, model_name):  # noqa: ARG001
        return [(labels[0], 0.91), (labels[1], 0.07), (labels[2], 0.02)]

    from cvsandbox.ai import hf_clip

    monkeypatch.setattr(hf_clip, "classify", _fake_classify)
    img = np.full((180, 360, 3), 150, dtype=np.uint8)
    streaming.set_current_node("nClip")
    try:
        ai_op.authorize_node("nClip")
        out = _run_clip(img, top_k=2)
    finally:
        streaming.set_current_node(None)
    # Image passes through unchanged — the reply lives in the side panel.
    assert np.array_equal(out, img)
    assert (
        streaming.get_node_display("nClip")
        == "a photo of a cat 0.91, a photo of a dog 0.07"
    )

    key = ai_op._clip_cache_key(
        img,
        ("a photo of a cat", "a photo of a dog", "a photo of a car"),
        "openai/clip-vit-base-patch32",
    )
    cached = ai_op._clip_cache_get(key)
    assert cached == "a photo of a cat 0.91, a photo of a dog 0.07"


def test_clip_classify_caches_extras_missing_error(
    sync_threads, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cvsandbox.ai import hf_clip

    def _boom(*_args, **_kwargs):
        raise hf_clip.HFExtrasMissing("install transformers + torch")

    monkeypatch.setattr(hf_clip, "classify", _boom)
    img = np.full((180, 360, 3), 200, dtype=np.uint8)
    _run_clip(img)
    # Re-run with the same image — must hit the cache (no second call).
    calls: list[int] = []

    def _count_call(*_args, **_kwargs):
        calls.append(1)
        return []

    monkeypatch.setattr(hf_clip, "classify", _count_call)
    _run_clip(img)
    assert calls == []  # cached error blocked retries

    key = ai_op._clip_cache_key(
        img,
        ("a photo of a cat", "a photo of a dog", "a photo of a car"),
        "openai/clip-vit-base-patch32",
    )
    cached = ai_op._clip_cache_get(key)
    assert cached is not None and "Setup" in cached


def test_clip_classify_label_change_creates_new_key(
    sync_threads, monkeypatch: pytest.MonkeyPatch
) -> None:
    received: list[tuple[str, ...]] = []

    def _fake_classify(image, labels, *, model_name):  # noqa: ARG001
        received.append(tuple(labels))
        return [(label, 1.0 / len(labels)) for label in labels]

    from cvsandbox.ai import hf_clip

    monkeypatch.setattr(hf_clip, "classify", _fake_classify)
    img = np.full((80, 80, 3), 100, dtype=np.uint8)
    _run_clip(img, labels="cat, dog")
    _run_clip(img, labels="ship, plane")
    assert received == [("cat", "dog"), ("ship", "plane")]


# --------------------------------------------------------------- OWL-ViT op


def _run_owlvit(img: np.ndarray, **overrides) -> np.ndarray:
    from cvsandbox.operations.ai import OWLVIT_DETECT

    params = {
        "prompts": "a photo of a person, a photo of a car",
        "model_name": "google/owlvit-base-patch32",
        "score_threshold": 0.1,
        "box_thickness": 2,
        "font_scale": 0.5,
    }
    params.update(overrides)
    return OWLVIT_DETECT.func(img, **params)


def test_color_for_label_is_deterministic_across_calls() -> None:
    # sha1 is stable across processes, so the same label always maps to
    # the same palette colour — verify both same-call determinism and
    # palette membership (we can't assert *distinct* colours since the
    # palette only has 6 entries and any two labels could collide).
    a1 = ai_op._color_for_label("person")
    a2 = ai_op._color_for_label("person")
    assert a1 == a2
    assert a1 in ai_op._DETECT_PALETTE


def test_draw_boxes_renders_rectangle_on_canvas() -> None:
    from cvsandbox.ai.hf_owlvit import Detection

    img = np.full((200, 320, 3), 128, dtype=np.uint8)
    detections = [Detection(label="person", score=0.84, box=(50, 60, 200, 180))]
    out = ai_op._draw_boxes(img, detections, font_scale=0.5, box_thickness=2)
    # Some pixels along the rectangle perimeter should now have the chosen
    # color rather than the uniform 128.
    diff = (out != 128).any(axis=2)
    assert diff[60:62, 50:200].any()  # top edge
    assert diff[179:181, 50:200].any()  # bottom edge


def test_draw_boxes_empty_detections_returns_unchanged_canvas() -> None:
    """No detections → no boxes, no banner — the side panel shows the
    "No objects matched" message; the image is left clean for any
    downstream OpenCV op."""
    img = np.full((200, 320, 3), 200, dtype=np.uint8)
    out = ai_op._draw_boxes(img, [], font_scale=0.6, box_thickness=2)
    assert out.shape == img.shape
    assert np.array_equal(out, img)


def test_draw_boxes_clips_out_of_bounds_box() -> None:
    from cvsandbox.ai.hf_owlvit import Detection

    img = np.full((100, 100, 3), 0, dtype=np.uint8)
    # Box partially outside the image — must not raise.
    detections = [Detection(label="x", score=0.5, box=(50, 50, 500, 500))]
    out = ai_op._draw_boxes(img, detections, font_scale=0.5, box_thickness=2)
    assert out.shape == img.shape


def test_owlvit_detect_caches_detection_list(
    sync_threads, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cvsandbox.ai import hf_owlvit
    from cvsandbox.ai.hf_owlvit import Detection

    calls: list[list[str]] = []

    def _fake_detect(image, prompts, *, model_name, score_threshold):  # noqa: ARG001
        calls.append(list(prompts))
        return [Detection(label=prompts[0], score=0.9, box=(10, 10, 80, 80))]

    monkeypatch.setattr(hf_owlvit, "detect", _fake_detect)
    img = np.full((120, 240, 3), 150, dtype=np.uint8)
    _run_owlvit(img)
    _run_owlvit(img)
    assert len(calls) == 1
    key = ai_op._detect_cache_key(
        img,
        ("a photo of a person", "a photo of a car"),
        "google/owlvit-base-patch32",
        0.1,
    )
    cached = ai_op._detect_cache_get(key)
    assert isinstance(cached, list)
    assert cached[0].label == "a photo of a person"


def test_owlvit_detect_requires_auth_in_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _should_not_spawn(_fn):
        raise AssertionError("OWL-ViT must not spawn without authorization")

    monkeypatch.setattr(ai_op, "_run_in_thread", _should_not_spawn)
    img = np.full((180, 360, 3), 200, dtype=np.uint8)
    streaming.set_current_node("nOwl")
    try:
        out = _run_owlvit(img)
    finally:
        streaming.set_current_node(None)
    assert out.shape == img.shape


def test_owlvit_detect_renders_boxes_for_cached_result(
    sync_threads, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cvsandbox.ai import hf_owlvit
    from cvsandbox.ai.hf_owlvit import Detection

    def _fake_detect(image, prompts, *, model_name, score_threshold):  # noqa: ARG001
        return [
            Detection(label=prompts[0], score=0.95, box=(30, 40, 200, 180)),
            Detection(label=prompts[1], score=0.62, box=(220, 50, 320, 170)),
        ]

    monkeypatch.setattr(hf_owlvit, "detect", _fake_detect)
    img = np.full((220, 360, 3), 150, dtype=np.uint8)
    out = _run_owlvit(img)
    assert out.shape == img.shape
    # The rendered image must differ from the input — boxes drawn.
    assert not np.array_equal(out, img)


def test_owlvit_detect_caches_extras_missing(
    sync_threads, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cvsandbox.ai import hf_owlvit
    from cvsandbox.ai.hf_clip import HFExtrasMissing

    def _boom(*_args, **_kwargs):
        raise HFExtrasMissing("install transformers + torch")

    monkeypatch.setattr(hf_owlvit, "detect", _boom)
    img = np.full((180, 360, 3), 200, dtype=np.uint8)
    _run_owlvit(img)

    calls: list[int] = []

    def _count_call(*_args, **_kwargs):
        calls.append(1)
        return []

    monkeypatch.setattr(hf_owlvit, "detect", _count_call)
    _run_owlvit(img)
    assert calls == []  # cached error blocked retries

    key = ai_op._detect_cache_key(
        img,
        ("a photo of a person", "a photo of a car"),
        "google/owlvit-base-patch32",
        0.1,
    )
    cached = ai_op._detect_cache_get(key)
    assert isinstance(cached, str) and "Setup" in cached


def test_owlvit_detect_empty_prompts_show_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cvsandbox.ai import hf_owlvit

    def _should_not_detect(*_args, **_kwargs):
        raise AssertionError("must not call hf_owlvit.detect without prompts")

    monkeypatch.setattr(hf_owlvit, "detect", _should_not_detect)
    img = np.full((80, 240, 3), 100, dtype=np.uint8)
    out = _run_owlvit(img, prompts="")
    assert out.shape == img.shape


def test_owlvit_detect_threshold_in_cache_key(
    sync_threads, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Changing the score threshold must invalidate the cache and trigger
    a fresh detection — it changes what OWL-ViT returns."""
    from cvsandbox.ai import hf_owlvit
    from cvsandbox.ai.hf_owlvit import Detection

    seen_thresholds: list[float] = []

    def _fake_detect(image, prompts, *, model_name, score_threshold):  # noqa: ARG001
        seen_thresholds.append(float(score_threshold))
        return [Detection(label=prompts[0], score=0.9, box=(10, 10, 80, 80))]

    monkeypatch.setattr(hf_owlvit, "detect", _fake_detect)
    img = np.full((100, 100, 3), 100, dtype=np.uint8)
    _run_owlvit(img, score_threshold=0.1)
    _run_owlvit(img, score_threshold=0.4)
    assert seen_thresholds == [0.1, 0.4]


# --------------------------------------------------------------- BLIP-2 op


def _run_blip2(img: np.ndarray, **overrides) -> np.ndarray:
    from cvsandbox.operations.ai import BLIP2_CAPTION

    params = {
        "model_name": "Salesforce/blip2-opt-2.7b",
        "max_new_tokens": 30,
    }
    params.update(overrides)
    return BLIP2_CAPTION.func(img, **params)


def test_blip2_caption_publishes_to_display_and_caches(
    sync_threads, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cvsandbox.ai import hf_blip2

    calls: list[int] = []

    def _fake_caption(image, *, model_name, max_new_tokens):  # noqa: ARG001
        calls.append(1)
        return "a brown dog sitting on grass"

    monkeypatch.setattr(hf_blip2, "caption", _fake_caption)
    img = np.full((180, 240, 3), 150, dtype=np.uint8)
    streaming.set_current_node("nBlip")
    try:
        ai_op.authorize_node("nBlip")
        out = _run_blip2(img)
        # Second call hits cache, no extra caption() invocation.
        _run_blip2(img)
    finally:
        streaming.set_current_node(None)

    assert np.array_equal(out, img)  # passthrough
    assert streaming.get_node_display("nBlip") == "a brown dog sitting on grass"
    assert len(calls) == 1
    key = ai_op._caption_cache_key(img, "Salesforce/blip2-opt-2.7b", 30)
    assert ai_op._caption_cache_get(key) == "a brown dog sitting on grass"


def test_blip2_caption_requires_auth_in_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _should_not_spawn(_fn):
        raise AssertionError("BLIP-2 must not spawn without authorization")

    monkeypatch.setattr(ai_op, "_run_in_thread", _should_not_spawn)
    img = np.full((100, 100, 3), 100, dtype=np.uint8)
    streaming.set_current_node("nB")
    try:
        out = _run_blip2(img)
    finally:
        streaming.set_current_node(None)
    assert np.array_equal(out, img)
    assert "Run" in (streaming.get_node_display("nB") or "")


def test_blip2_caption_caches_extras_missing(
    sync_threads, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cvsandbox.ai import hf_blip2
    from cvsandbox.ai.hf_clip import HFExtrasMissing

    def _boom(*_args, **_kwargs):
        raise HFExtrasMissing("install transformers + torch")

    monkeypatch.setattr(hf_blip2, "caption", _boom)
    img = np.full((60, 60, 3), 90, dtype=np.uint8)
    _run_blip2(img)
    key = ai_op._caption_cache_key(img, "Salesforce/blip2-opt-2.7b", 30)
    cached = ai_op._caption_cache_get(key)
    assert cached is not None and "Setup" in cached


def test_blip2_caption_tokens_in_cache_key(
    sync_threads, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Changing max_new_tokens must invalidate the cache so the user
    sees the longer caption they asked for."""
    from cvsandbox.ai import hf_blip2

    seen: list[int] = []

    def _fake_caption(image, *, model_name, max_new_tokens):  # noqa: ARG001
        seen.append(int(max_new_tokens))
        return "short" if max_new_tokens <= 20 else "much longer caption text"

    monkeypatch.setattr(hf_blip2, "caption", _fake_caption)
    img = np.full((80, 80, 3), 100, dtype=np.uint8)
    _run_blip2(img, max_new_tokens=20)
    _run_blip2(img, max_new_tokens=80)
    assert seen == [20, 80]


# --------------------------------------------------------------- auto_run


def test_auto_run_bypasses_authorization_gate(
    sync_threads, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With auto_run=True the op must spawn even when the user hasn't
    pressed Run — this is what makes per-frame AI on a video source
    possible."""
    calls: list[str] = []

    def _fake_stream(prompt, image, **_kwargs) -> Iterator[str]:  # noqa: ARG001
        calls.append(prompt)
        yield "auto reply"

    monkeypatch.setattr(ai_op, "stream_generate", _fake_stream)
    img = np.full((80, 80, 3), 100, dtype=np.uint8)
    streaming.set_current_node("nAuto")
    try:
        # No authorize_node call — auto_run should still drive the worker.
        _run_op(img, auto_run=True)
    finally:
        streaming.set_current_node(None)
    assert calls == ["What is this?"]


def test_auto_run_off_still_requires_authorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default `auto_run=False` keeps the manual-Run discipline."""

    def _should_not_spawn(_fn):
        raise AssertionError("auto_run=False must keep the auth gate")

    monkeypatch.setattr(ai_op, "_run_in_thread", _should_not_spawn)
    img = np.full((80, 80, 3), 100, dtype=np.uint8)
    streaming.set_current_node("nManual")
    try:
        _run_op(img, auto_run=False)
    finally:
        streaming.set_current_node(None)


def test_every_ai_op_exposes_auto_run_param() -> None:
    """Smoke-check: each AI op spec must declare auto_run so the
    parameter panel exposes a single, uniform "run on every frame"
    toggle across backends."""
    from cvsandbox.operations.ai import (
        BLIP2_CAPTION,
        CLIP_CLASSIFY,
        OWLVIT_DETECT,
        VLM_QUERY,
    )

    for spec in (VLM_QUERY, CLIP_CLASSIFY, OWLVIT_DETECT, BLIP2_CAPTION):
        names = [p.name for p in spec.parameters]
        assert "auto_run" in names, f"{spec.id} is missing the auto_run param"
        param = next(p for p in spec.parameters if p.name == "auto_run")
        assert param.kind == "bool"
        assert param.default is False


def test_pipeline_worker_sets_current_node_around_each_call() -> None:
    """The worker's thread-local plumbing is what makes
    `cancel_node_streams_except` meaningful — verify each step sees its
    node_id and that the local is cleared between steps."""
    from cvsandbox.ui.pipeline_worker import PipelineWorker

    observed: list[str | None] = []

    def _record(image: np.ndarray, **_params: object) -> np.ndarray:
        observed.append(streaming.current_node())
        return image

    image = np.zeros((4, 4), dtype=np.uint8)
    steps = ((_record, {}, "nA"), (_record, {}, "nB"))
    PipelineWorker._run_steps(image, steps)  # type: ignore[arg-type]
    assert observed == ["nA", "nB"]
    # After the loop, the thread-local must be cleared.
    assert streaming.current_node() is None


# --------------------------------------------------------------- vlm_query op


def _run_op(img: np.ndarray, **overrides) -> np.ndarray:
    params = {
        "prompt": "What is this?",
        "model": "llava",
        "host": "http://localhost:11434",
        "temperature": 0.2,
        "font_scale": 0.6,
    }
    params.update(overrides)
    return VLM_QUERY.func(img, **params)


def _patch_stream(monkeypatch: pytest.MonkeyPatch, tokens: list[str]) -> list[str]:
    """Replace `stream_generate` with a generator yielding `tokens`. Returns
    a list that will be populated with each prompt the worker received."""
    captured_prompts: list[str] = []

    def _fake_stream(prompt, image, **_kwargs) -> Iterator[str]:  # noqa: ARG001
        captured_prompts.append(prompt)
        yield from tokens

    monkeypatch.setattr(ai_op, "stream_generate", _fake_stream)
    return captured_prompts


def test_vlm_query_caches_final_text_after_stream(
    sync_threads, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_stream(monkeypatch, ["A", " red", " square"])
    img = np.full((120, 240, 3), 150, dtype=np.uint8)
    out = _run_op(img)
    assert out.shape == img.shape
    # In sync mode the worker has already finished; cache holds the joined text.
    key = ai_op._cache_key(img, "What is this?", "llava", 0.2)
    assert ai_op._cache_get(key) == "A red square"


def test_vlm_query_partial_visible_during_stream_in_display(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Async (no sync_threads): manually mark a stream active with a
    partial, then assert the op publishes the partial text via the
    per-node display (not the image)."""
    img = np.full((180, 360, 3), 200, dtype=np.uint8)
    key = ai_op._cache_key(img, "What is this?", "llava", 0.2)
    streaming.begin_stream(key)
    streaming.append_token(key, "Hello, partial")

    def _should_not_spawn(*_args, **_kwargs):
        raise AssertionError("must not spawn a second stream for the same key")

    monkeypatch.setattr(ai_op, "_run_in_thread", _should_not_spawn)

    streaming.set_current_node("nPartial")
    try:
        out = _run_op(img)
    finally:
        streaming.set_current_node(None)
    assert np.array_equal(out, img)  # image untouched
    text = streaming.get_node_display("nPartial") or ""
    assert "Hello, partial" in text


def test_vlm_query_thinking_placeholder_published_to_display(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    img = np.full((180, 360, 3), 200, dtype=np.uint8)
    spawned: list[int] = []

    def _swallow_spawn(_fn):
        spawned.append(1)

    monkeypatch.setattr(ai_op, "_run_in_thread", _swallow_spawn)
    streaming.set_current_node("nThink")
    try:
        ai_op.authorize_node("nThink")
        out = _run_op(img)
    finally:
        streaming.set_current_node(None)
    assert spawned == [1]
    assert np.array_equal(out, img)
    # Either the "Thinking…" placeholder or whatever the partial is — but
    # something must have been published.
    assert (streaming.get_node_display("nThink") or "").strip() != ""


def test_vlm_query_second_call_hits_cache(
    sync_threads, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompts = _patch_stream(monkeypatch, ["cached reply"])
    img = np.full((80, 80, 3), 100, dtype=np.uint8)
    _run_op(img)
    _run_op(img)  # cache hit — must not invoke stream again
    _run_op(img, font_scale=1.2)  # font_scale not in cache key — still cache hit
    assert prompts == ["What is this?"]


def test_vlm_query_re_invokes_on_prompt_change(
    sync_threads, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompts = _patch_stream(monkeypatch, ["reply"])
    img = np.full((80, 80, 3), 100, dtype=np.uint8)
    _run_op(img, prompt="What is this?")
    _run_op(img, prompt="Count the objects.")
    assert prompts == ["What is this?", "Count the objects."]


def test_vlm_query_caches_error_to_prevent_retry_storm(
    sync_threads, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []

    def _boom(*_args, **_kwargs) -> Iterator[str]:
        calls.append(1)
        raise ollama_client.OllamaError("offline")
        yield  # unreachable — keeps this a generator function

    monkeypatch.setattr(ai_op, "stream_generate", _boom)
    img = np.full((240, 480, 3), 200, dtype=np.uint8)
    streaming.set_current_node("nErr")
    try:
        ai_op.authorize_node("nErr")
        out = _run_op(img)
        _run_op(img)
        _run_op(img)
    finally:
        streaming.set_current_node(None)
    # First call drove the (failing) worker; subsequent calls saw the cached
    # error and never touched the backend again.
    assert len(calls) == 1
    key = ai_op._cache_key(img, "What is this?", "llava", 0.2)
    cached = ai_op._cache_get(key)
    assert cached is not None and "Ollama" in cached
    # Image stays clean — error message is in the side panel.
    assert np.array_equal(out, np.full_like(out, 200))
    assert "Ollama" in (streaming.get_node_display("nErr") or "")


def test_vlm_query_empty_prompt_does_not_spawn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _should_not_spawn(*_args, **_kwargs):
        raise AssertionError("empty prompt must short-circuit before spawning")

    monkeypatch.setattr(ai_op, "_run_in_thread", _should_not_spawn)
    img = np.full((40, 40, 3), 100, dtype=np.uint8)
    out = _run_op(img, prompt="   ")
    assert out.shape == img.shape


def test_vlm_query_passes_grayscale_through_unchanged(
    sync_threads, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With output text moved to the side panel, the op is a true
    passthrough — grayscale stays grayscale, BGR stays BGR, no
    surprising channel promotion for downstream nodes."""
    _patch_stream(monkeypatch, ["gray reply"])
    gray = np.full((60, 60), 128, dtype=np.uint8)
    out = _run_op(gray)
    assert out.shape == gray.shape
    assert out.ndim == 2
