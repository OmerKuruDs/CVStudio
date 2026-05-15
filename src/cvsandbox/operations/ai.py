"""AI operations — pipeline nodes that delegate to a vision-language /
classification / detection / captioning backend.

All four current ops follow the same shape: per-image cache keyed by
the model-affecting params, one-shot Run authorization, daemon-thread
inference, and a textual reply that lands in the right-side AI Response
panel via `streaming.set_node_display`. The shared scaffolding lives
in `cvsandbox.ai.backend.AIBackend`; the concrete classes below only
supply the bits that differ — validation, cache key shape, the
inference call itself, and (for OWL-ViT) the on-image box overlay.

Test compatibility:

The pre-refactor test suite reached into module-level helpers like
`_cache_key`, `_cache_put`, `_color_for_label`, etc. The compatibility
shims at the bottom of this file forward those names to the new
backend objects so existing tests keep passing — they're intentional
public-ish API for the test layer rather than dead code.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Iterator
from typing import Any

import cv2
import numpy as np

from cvsandbox.ai import hf_blip2, hf_clip, hf_owlvit, streaming
from cvsandbox.ai.backend import (
    AIBackend,
    StreamingAIBackend,
    authorize_node,
    clear_authorizations,
    consume_auth as _backend_consume_auth,
    run_in_thread as _default_run_in_thread,
)


def _run_in_thread(target: Any) -> None:
    """Module-level indirection so test code can monkeypatch
    `ai_op._run_in_thread` and have every AI backend pick up the
    synchronous runner. The backends are wired through `_proxy_runner`
    below so this name is the single point of replacement."""
    _default_run_in_thread(target)


def _proxy_runner(target: Any) -> None:
    """Re-resolve `_run_in_thread` at call time so monkeypatching the
    module global flows through to every backend instance."""
    _run_in_thread(target)
from cvsandbox.ai.hf_clip import HFExtrasMissing
from cvsandbox.ai.hf_owlvit import Detection
from cvsandbox.ai.ollama_client import (
    DEFAULT_HOST,
    DEFAULT_MODEL,
    OllamaError,
    stream_generate,
)
from cvsandbox.core.operation import OperationSpec, Parameter

# --- shared helpers used by multiple backends ----------------------------

_AUTO_RUN_DESCRIPTION = (
    "When ON, skips the Run button and authorizes every pipeline pass — "
    "useful for video sources. Auto-cancellation prevents requests from "
    "piling up when the model is slower than the frame rate."
)


def _parse_labels(raw: str) -> list[str]:
    """Split a comma- / semicolon- / newline-separated label string
    into a clean list. Empty entries dropped, whitespace trimmed."""
    chunks: list[str] = []
    for line in raw.replace(";", ",").splitlines():
        for token in line.split(","):
            label = token.strip()
            if label:
                chunks.append(label)
    return chunks


def _image_hash(image: np.ndarray) -> str:
    return hashlib.sha1(image.tobytes()).hexdigest()


# ========================================================== VLM (Ollama) ==


class VLMBackend(StreamingAIBackend):
    name = "vlm"

    def validate(self, params: dict[str, Any]) -> str | None:
        if not (params.get("prompt") or "").strip():
            return "(set a prompt in the parameter panel)"
        return None

    def make_key(self, image: np.ndarray, params: dict[str, Any]) -> tuple:
        return (
            _image_hash(image),
            str(params["prompt"]).strip(),
            str(params["model"]),
            round(float(params["temperature"]), 3),
        )

    def iter_tokens(
        self, image: np.ndarray, params: dict[str, Any]
    ) -> Iterator[str]:
        # `stream_generate` is module-level so tests can monkeypatch it.
        yield from stream_generate(
            str(params["prompt"]).strip(),
            image,
            model=str(params["model"]),
            host=str(params["host"]),
            temperature=float(params["temperature"]),
        )

    def format_error(self, exc: Exception) -> str | None:
        if isinstance(exc, OllamaError):
            return f"[Ollama] {exc}"
        return None


_vlm = VLMBackend()


def _vlm_query(
    image: np.ndarray,
    prompt: str,
    model: str,
    host: str,
    temperature: float,
    font_scale: float,
    auto_run: bool = False,
) -> np.ndarray:
    """Image passes through unchanged; the reply (or error / pending
    hint) lands in the per-node display store."""
    del font_scale  # legacy param, kept for saved-pipeline compatibility
    return _vlm.execute(
        image,
        prompt=prompt,
        model=model,
        host=host,
        temperature=temperature,
        auto_run=auto_run,
    )


def _vlm_query_code(
    params: dict[str, Any], input_vars: tuple[str, ...], output_var: str
) -> list[str]:
    (a,) = input_vars
    prompt = repr(params.get("prompt", ""))
    model = repr(params.get("model", ""))
    return [
        "# VLM Q&A (Ollama) — not reproduced in exported code.",
        f"# Prompt: {prompt}",
        f"# Model:  {model}",
        f"# Skipping the Ollama call; downstream sees the unmodified image.",
        f"{output_var} = {a}",
    ]


VLM_QUERY = OperationSpec(
    id="ai.vlm_query",
    name="VLM Q&A (Ollama)",
    category="AI",
    description=(
        "Send the current image to a local Ollama vision-language model with a "
        "prompt and stream the reply into the AI Response panel. Requires "
        "Ollama running locally (default http://localhost:11434) with a VLM "
        "(e.g. `ollama pull llava`)."
    ),
    parameters=(
        Parameter(
            name="prompt",
            kind="string",
            default="Describe this image in one short sentence.",
            step=3,
            label="Prompt",
            description="Question or instruction sent to the model.",
        ),
        Parameter(
            name="model",
            kind="string",
            default=DEFAULT_MODEL,
            label="Model",
            description="Ollama model tag (e.g. llava, llava:13b, bakllava).",
        ),
        Parameter(
            name="host",
            kind="string",
            default=DEFAULT_HOST,
            label="Host",
            description="Ollama server URL.",
        ),
        Parameter(
            name="temperature",
            kind="float",
            default=0.2,
            min=0.0,
            max=1.5,
            step=0.05,
            label="Temperature",
            description="Higher = more creative, lower = more deterministic.",
        ),
        Parameter(
            name="font_scale",
            kind="float",
            default=0.6,
            min=0.3,
            max=1.5,
            step=0.05,
            label="Banner font scale",
            description=(
                "Legacy: kept so older saved pipelines still load. The "
                "response now lives in the side panel, not the image."
            ),
        ),
        Parameter(
            name="auto_run",
            kind="bool",
            default=False,
            label="Auto-run (video)",
            description=_AUTO_RUN_DESCRIPTION,
        ),
    ),
    func=_vlm_query,
    code_export=_vlm_query_code,
    manual_trigger=True,
)


# ============================================================ CLIP classify ==


def _format_clip_result(pairs: list[tuple[str, float]], top_k: int) -> str:
    if not pairs:
        return "(no labels — set them in the parameter panel)"
    selected = pairs[: max(1, int(top_k))]
    return ", ".join(f"{label} {score:.2f}" for label, score in selected)


class CLIPBackend(AIBackend):
    name = "clip"
    running_placeholder = "Classifying…"

    def validate(self, params: dict[str, Any]) -> str | None:
        labels = _parse_labels(params.get("labels", "") or "")
        if not labels:
            return "(add comma-separated labels in the parameter panel)"
        return None

    def make_key(self, image: np.ndarray, params: dict[str, Any]) -> tuple:
        labels = tuple(_parse_labels(params.get("labels", "") or ""))
        return (_image_hash(image), labels, str(params["model_name"]))

    def run(
        self,
        key: tuple,
        image: np.ndarray,
        params: dict[str, Any],
        node_id: str | None,
    ) -> str | None:
        del key
        streaming.set_node_display(node_id, "Loading model…")
        labels = _parse_labels(params.get("labels", "") or "")
        pairs = hf_clip.classify(
            image, labels, model_name=str(params["model_name"])
        )
        return _format_clip_result(pairs, int(params["top_k"]))

    def format_display(self, result: str) -> str:
        return result


_clip = CLIPBackend()


def _clip_classify(
    image: np.ndarray,
    labels: str,
    model_name: str,
    top_k: int,
    font_scale: float,
    auto_run: bool = False,
) -> np.ndarray:
    del font_scale
    return _clip.execute(
        image,
        labels=labels,
        model_name=model_name,
        top_k=top_k,
        auto_run=auto_run,
    )


def _clip_classify_code(
    params: dict[str, Any], input_vars: tuple[str, ...], output_var: str
) -> list[str]:
    (a,) = input_vars
    labels = repr(params.get("labels", ""))
    model = repr(params.get("model_name", ""))
    return [
        "# CLIP zero-shot classify — not reproduced in exported code.",
        f"# Labels: {labels}",
        f"# Model:  {model}",
        f"{output_var} = {a}",
    ]


CLIP_CLASSIFY = OperationSpec(
    id="ai.clip_classify",
    name="CLIP Zero-shot Classify",
    category="AI",
    description=(
        "Score the image against comma-separated text labels using OpenAI's "
        "CLIP. Top-K labels with similarity scores land in the AI Response "
        "panel. Requires the optional `[ai]` extras: pip install -e .[ai]."
    ),
    parameters=(
        Parameter(
            name="labels",
            kind="string",
            default="a photo of a cat, a photo of a dog, a photo of a car",
            step=3,
            label="Labels (comma-separated)",
        ),
        Parameter(
            name="model_name",
            kind="string",
            default=hf_clip.DEFAULT_MODEL,
            label="HF model name",
        ),
        Parameter(
            name="top_k",
            kind="int",
            default=3,
            min=1,
            max=10,
            label="Top-K results",
        ),
        Parameter(
            name="font_scale",
            kind="float",
            default=0.6,
            min=0.3,
            max=1.5,
            step=0.05,
            label="Banner font scale",
            description="Legacy — output now in side panel.",
        ),
        Parameter(
            name="auto_run",
            kind="bool",
            default=False,
            label="Auto-run (video)",
            description=_AUTO_RUN_DESCRIPTION,
        ),
    ),
    func=_clip_classify,
    code_export=_clip_classify_code,
    manual_trigger=True,
)


# ============================================================ OWL-ViT detect ==


_DETECT_PALETTE: tuple[tuple[int, int, int], ...] = (
    (0, 200, 0),       # green
    (50, 50, 230),     # red-ish
    (230, 100, 50),    # blue
    (0, 200, 230),     # yellow
    (230, 50, 200),    # magenta
    (200, 200, 50),    # cyan
)


def _color_for_label(label: str) -> tuple[int, int, int]:
    """Pick a palette colour deterministically from `label`. sha1 (not
    Python's salted built-in `hash`) so the same prompt maps to the
    same colour across app restarts."""
    digest = hashlib.sha1(label.encode("utf-8")).digest()
    return _DETECT_PALETTE[digest[0] % len(_DETECT_PALETTE)]


def _draw_boxes(
    image: np.ndarray,
    detections: list[Detection],
    font_scale: float,
    box_thickness: int,
) -> np.ndarray:
    canvas = image.copy()
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    elif canvas.shape[2] == 4:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGRA2BGR)

    if not detections:
        # Caller (OWL-ViT op) decides what to do with an empty list —
        # we just return the unchanged canvas so the side-panel "No
        # objects matched" message is the sole feedback.
        return canvas

    h, w = canvas.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_thickness = max(1, int(round(font_scale * 1.4)))

    for det in detections:
        x1, y1, x2, y2 = det.box
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        color = _color_for_label(det.label)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, int(box_thickness))

        chip_text = f"{det.label} {det.score:.2f}"
        (tw, th), baseline = cv2.getTextSize(
            chip_text, font, float(font_scale), text_thickness
        )
        pad = 3
        chip_h = th + baseline + pad * 2
        if y1 - chip_h >= 0:
            chip_y1, chip_y2 = y1 - chip_h, y1
        else:
            chip_y1, chip_y2 = y1, min(h, y1 + chip_h)
        chip_x1 = x1
        chip_x2 = min(w, x1 + tw + pad * 2)
        cv2.rectangle(canvas, (chip_x1, chip_y1), (chip_x2, chip_y2), color, thickness=-1)
        cv2.putText(
            canvas,
            chip_text,
            (chip_x1 + pad, chip_y2 - baseline - pad // 2),
            font,
            float(font_scale),
            (255, 255, 255),
            text_thickness,
            cv2.LINE_AA,
        )
    return canvas


def _format_detections_summary(detections: list[Detection]) -> str:
    if not detections:
        return "No objects matched the prompts."
    lines = [f"{len(detections)} detection(s):"]
    for det in detections:
        x1, y1, x2, y2 = det.box
        lines.append(
            f"  • {det.label} ({det.score:.2f}) at [{x1}, {y1}, {x2}, {y2}]"
        )
    return "\n".join(lines)


class OWLViTBackend(AIBackend):
    name = "owlvit"
    running_placeholder = "Detecting…"

    def validate(self, params: dict[str, Any]) -> str | None:
        prompts = _parse_labels(params.get("prompts", "") or "")
        if not prompts:
            return "(add comma-separated prompts in the parameter panel)"
        return None

    def make_key(self, image: np.ndarray, params: dict[str, Any]) -> tuple:
        prompts = tuple(_parse_labels(params.get("prompts", "") or ""))
        return (
            _image_hash(image),
            prompts,
            str(params["model_name"]),
            round(float(params["score_threshold"]), 3),
        )

    def run(
        self,
        key: tuple,
        image: np.ndarray,
        params: dict[str, Any],
        node_id: str | None,
    ) -> list[Detection] | None:
        del key
        streaming.set_node_display(node_id, "Loading model…")
        prompts = _parse_labels(params.get("prompts", "") or "")
        return hf_owlvit.detect(
            image,
            prompts,
            model_name=str(params["model_name"]),
            score_threshold=float(params["score_threshold"]),
        )

    def format_display(self, result: list[Detection]) -> str:
        return _format_detections_summary(result)

    def render(
        self,
        image: np.ndarray,
        result: list[Detection],
        params: dict[str, Any],
    ) -> np.ndarray:
        if not result:
            return image  # No boxes to draw; side panel shows the empty message.
        return _draw_boxes(
            image,
            result,
            float(params["font_scale"]),
            int(params["box_thickness"]),
        )

    # Persist Detection lists as plain dicts so the on-disk cache is
    # readable / forward-compatible (the dataclass is internal).
    def serialize_value(self, value: list[Detection] | str) -> Any:
        if isinstance(value, str):
            return {"kind": "error", "text": value}
        return {
            "kind": "detections",
            "items": [
                {"label": d.label, "score": d.score, "box": list(d.box)}
                for d in value
            ],
        }

    def deserialize_value(self, raw: Any) -> list[Detection] | str | None:
        if not isinstance(raw, dict):
            return None
        kind = raw.get("kind")
        if kind == "error":
            text = raw.get("text", "")
            return str(text) if text else None
        if kind == "detections":
            out: list[Detection] = []
            for item in raw.get("items", []) or []:
                try:
                    box = item["box"]
                    out.append(
                        Detection(
                            label=str(item["label"]),
                            score=float(item["score"]),
                            box=(int(box[0]), int(box[1]), int(box[2]), int(box[3])),
                        )
                    )
                except (KeyError, TypeError, ValueError):
                    continue
            return out
        return None


_owlvit = OWLViTBackend()


def _owlvit_detect(
    image: np.ndarray,
    prompts: str,
    model_name: str,
    score_threshold: float,
    box_thickness: int,
    font_scale: float,
    auto_run: bool = False,
) -> np.ndarray:
    return _owlvit.execute(
        image,
        prompts=prompts,
        model_name=model_name,
        score_threshold=score_threshold,
        box_thickness=box_thickness,
        font_scale=font_scale,
        auto_run=auto_run,
    )


def _owlvit_detect_code(
    params: dict[str, Any], input_vars: tuple[str, ...], output_var: str
) -> list[str]:
    (a,) = input_vars
    prompts = repr(params.get("prompts", ""))
    model = repr(params.get("model_name", ""))
    return [
        "# OWL-ViT zero-shot detection — not reproduced in exported code.",
        f"# Prompts: {prompts}",
        f"# Model:   {model}",
        f"{output_var} = {a}",
    ]


OWLVIT_DETECT = OperationSpec(
    id="ai.owlvit_detect",
    name="OWL-ViT Zero-shot Detection",
    category="AI",
    description=(
        "Find objects matching text prompts using Google's OWL-ViT. Draws "
        "colored bounding boxes (one color per prompt) on the image; a "
        "per-detection summary also lands in the AI Response panel. "
        "Requires the optional `[ai]` extras."
    ),
    parameters=(
        Parameter(
            name="prompts",
            kind="string",
            default="a photo of a person, a photo of a car",
            step=3,
            label="Prompts (comma-separated)",
        ),
        Parameter(
            name="model_name",
            kind="string",
            default=hf_owlvit.DEFAULT_MODEL,
            label="HF model name",
        ),
        Parameter(
            name="score_threshold",
            kind="float",
            default=0.1,
            min=0.01,
            max=1.0,
            step=0.01,
            label="Score threshold",
        ),
        Parameter(
            name="box_thickness",
            kind="int",
            default=2,
            min=1,
            max=8,
            label="Box line thickness",
        ),
        Parameter(
            name="font_scale",
            kind="float",
            default=0.5,
            min=0.3,
            max=1.5,
            step=0.05,
            label="Label font scale",
        ),
        Parameter(
            name="auto_run",
            kind="bool",
            default=False,
            label="Auto-run (video)",
            description=_AUTO_RUN_DESCRIPTION,
        ),
    ),
    func=_owlvit_detect,
    code_export=_owlvit_detect_code,
    manual_trigger=True,
)


# ============================================================ BLIP-2 caption ==


class BLIP2Backend(AIBackend):
    name = "blip2"
    running_placeholder = "Captioning…"

    def validate(self, params: dict[str, Any]) -> str | None:
        del params
        return None  # no required params beyond the defaults

    def make_key(self, image: np.ndarray, params: dict[str, Any]) -> tuple:
        return (
            _image_hash(image),
            str(params["model_name"]),
            int(params["max_new_tokens"]),
        )

    def run(
        self,
        key: tuple,
        image: np.ndarray,
        params: dict[str, Any],
        node_id: str | None,
    ) -> str | None:
        del key
        streaming.set_node_display(node_id, "Loading model…")
        return hf_blip2.caption(
            image,
            model_name=str(params["model_name"]),
            max_new_tokens=int(params["max_new_tokens"]),
        )

    def format_display(self, result: str) -> str:
        return result


_blip2 = BLIP2Backend()


def _blip2_caption(
    image: np.ndarray,
    model_name: str,
    max_new_tokens: int,
    auto_run: bool = False,
) -> np.ndarray:
    return _blip2.execute(
        image,
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        auto_run=auto_run,
    )


def _blip2_caption_code(
    params: dict[str, Any], input_vars: tuple[str, ...], output_var: str
) -> list[str]:
    (a,) = input_vars
    model = repr(params.get("model_name", ""))
    return [
        "# BLIP-2 captioning — not reproduced in exported code.",
        f"# Model: {model}",
        f"{output_var} = {a}",
    ]


BLIP2_CAPTION = OperationSpec(
    id="ai.blip2_caption",
    name="BLIP-2 Caption",
    category="AI",
    description=(
        "Generate a free-form caption for the image using HuggingFace's BLIP-2. "
        "Result lands in the AI Response panel — no prompt or labels needed. "
        "Requires the optional `[ai]` extras."
    ),
    parameters=(
        Parameter(
            name="model_name",
            kind="string",
            default=hf_blip2.DEFAULT_MODEL,
            label="HF model name",
        ),
        Parameter(
            name="max_new_tokens",
            kind="int",
            default=30,
            min=8,
            max=120,
            label="Max caption length (tokens)",
        ),
        Parameter(
            name="auto_run",
            kind="bool",
            default=False,
            label="Auto-run (video)",
            description=_AUTO_RUN_DESCRIPTION,
        ),
    ),
    func=_blip2_caption,
    code_export=_blip2_caption_code,
    manual_trigger=True,
)


ALL: tuple[OperationSpec, ...] = (VLM_QUERY, CLIP_CLASSIFY, OWLVIT_DETECT, BLIP2_CAPTION)


# ============================================================ public API ==
#
# Module-level helpers — used by MainWindow (cache persistence, clear-cache
# menu item) and by the test suite (direct cache poking).


def clear_cache() -> None:
    """Drop every cached generation, every streaming partial, and every
    pending node authorization. Surfaced in the UI as Tools → Clear AI
    cache so the user can force a fresh inference run."""
    for backend in _BACKENDS_BY_NAME.values():
        backend.cache_clear()
    clear_authorizations()
    streaming.reset()


_BACKENDS_BY_NAME: dict[str, AIBackend] = {
    "vlm": _vlm,
    "clip": _clip,
    "owlvit": _owlvit,
    "blip2": _blip2,
}

# Route every backend's spawn through `_proxy_runner` so monkeypatching
# `ai_op._run_in_thread` from tests changes how all four ops dispatch.
for _b in _BACKENDS_BY_NAME.values():
    _b.set_runner(_proxy_runner)


def all_backends() -> dict[str, AIBackend]:
    """Snapshot of the registered AI backends, keyed by their `name`
    attribute. Used by `cache_storage` to enumerate caches for
    serialization."""
    return dict(_BACKENDS_BY_NAME)


# ============================================================ test compat ==
#
# The pre-refactor test suite reached into module-level dict-style cache
# helpers. We expose those as forwarders to the per-backend cache so
# tests continue to pass without restructuring — and so anyone inspecting
# the codebase from outside can hit the same surface as before.


# VLM cache — `(image_hash, prompt, model, temperature)` → str
_VLM_CACHE: "OrderedDict[tuple, str]" = _vlm._cache  # type: ignore[assignment]
_CAPTION_CACHE: "OrderedDict[tuple, str]" = _blip2._cache  # type: ignore[assignment]
_DETECT_CACHE: "OrderedDict[tuple, list[Detection] | str]" = (
    _owlvit._cache  # type: ignore[assignment]
)


def _cache_key(
    image: np.ndarray, prompt: str, model: str, temperature: float
) -> tuple[str, str, str, float]:
    return (_image_hash(image), prompt, model, round(float(temperature), 3))


def _cache_get(key: tuple) -> str | None:
    cached = _vlm.cache_get(key)
    return cached if isinstance(cached, str) else None


def _cache_put(key: tuple, value: str) -> None:
    _vlm.cache_put(key, value)


def _clip_cache_key(
    image: np.ndarray, labels: tuple[str, ...], model_name: str
) -> tuple:
    return (_image_hash(image), labels, model_name)


def _clip_cache_get(key: tuple) -> str | None:
    cached = _clip.cache_get(key)
    return cached if isinstance(cached, str) else None


def _clip_cache_put(key: tuple, value: str) -> None:
    _clip.cache_put(key, value)


def _detect_cache_key(
    image: np.ndarray,
    prompts: tuple[str, ...],
    model_name: str,
    threshold: float,
) -> tuple:
    return (_image_hash(image), prompts, model_name, round(float(threshold), 3))


def _detect_cache_get(key: tuple) -> "list[Detection] | str | None":
    return _owlvit.cache_get(key)


def _detect_cache_put(key: tuple, value: "list[Detection] | str") -> None:
    _owlvit.cache_put(key, value)


def _caption_cache_key(
    image: np.ndarray, model_name: str, max_new_tokens: int
) -> tuple[str, str, int]:
    return (_image_hash(image), model_name, int(max_new_tokens))


def _caption_cache_get(key: tuple) -> str | None:
    cached = _blip2.cache_get(key)
    return cached if isinstance(cached, str) else None


def _caption_cache_put(key: tuple, value: str) -> None:
    _blip2.cache_put(key, value)


def _consume_auth(node_id: str | None) -> bool:
    """Compat shim — the new auth lives in `backend.consume_auth`."""
    return _backend_consume_auth(node_id)


__all__ = [
    "ALL",
    "BLIP2_CAPTION",
    "CLIP_CLASSIFY",
    "Detection",
    "HFExtrasMissing",
    "OWLVIT_DETECT",
    "OllamaError",
    "VLM_QUERY",
    "all_backends",
    "authorize_node",
    "clear_cache",
    "stream_generate",
]
