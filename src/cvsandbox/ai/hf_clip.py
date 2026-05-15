"""HuggingFace CLIP backend for zero-shot classification.

CLIP takes an image plus a list of candidate text labels and returns a
similarity score per label. We expose the simplest possible surface —
`classify(image, labels) -> [(label, score), ...]` sorted descending —
plus a friendly error when the optional `ai` extras are not installed.

Model + processor are cached at module level keyed by model name, so the
first call eats a multi-second model load and subsequent calls (with
the same model) are fast. Inference is CPU by default; users with CUDA
will benefit automatically because torch picks GPU when available.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


class HFExtrasMissing(RuntimeError):
    """Raised when the user runs an HF op without `pip install -e .[ai]`.

    The op surfaces this as a banner so the user sees the fix without
    having to read the traceback."""


@dataclass(frozen=True)
class _CachedModel:
    model: object  # transformers.CLIPModel — typed as object to avoid hard dep at module load
    processor: object  # transformers.CLIPProcessor
    device: str


_models: dict[str, _CachedModel] = {}

DEFAULT_MODEL = "openai/clip-vit-base-patch32"


def _ensure_imports() -> tuple[object, object, object]:
    """Lazy import. Raises HFExtrasMissing with a fix-it message when the
    `ai` extras aren't on the path so users see something more helpful
    than a raw `ModuleNotFoundError`."""
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as exc:
        raise HFExtrasMissing(
            "HuggingFace extras not installed. "
            "Run: pip install -e .[ai]  (adds transformers + torch + pillow)"
        ) from exc
    return torch, CLIPModel, CLIPProcessor


def _load_model(model_name: str) -> _CachedModel:
    if model_name in _models:
        return _models[model_name]
    torch, CLIPModel, CLIPProcessor = _ensure_imports()  # type: ignore[misc]
    device = "cuda" if torch.cuda.is_available() else "cpu"  # type: ignore[attr-defined]
    model = CLIPModel.from_pretrained(model_name).to(device)  # type: ignore[attr-defined]
    model.eval()  # type: ignore[attr-defined]
    processor = CLIPProcessor.from_pretrained(model_name)  # type: ignore[attr-defined]
    cached = _CachedModel(model=model, processor=processor, device=device)
    _models[model_name] = cached
    return cached


def _to_rgb_array(image: np.ndarray) -> np.ndarray:
    """CLIP's processor wants H×W×3 RGB. OpenCV gives us BGR (or grayscale,
    or BGRA) — coerce here so the op contract stays uniform."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def classify(
    image: np.ndarray,
    labels: list[str],
    *,
    model_name: str = DEFAULT_MODEL,
) -> list[tuple[str, float]]:
    """Run zero-shot classification. Returns `(label, score)` pairs sorted
    by score descending. `score` is the softmax probability across the
    label set so the values sum to ~1.0."""
    if not labels:
        return []

    torch, _, _ = _ensure_imports()
    cached = _load_model(model_name)

    rgb = _to_rgb_array(image)
    inputs = cached.processor(  # type: ignore[attr-defined]
        text=labels,
        images=rgb,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(cached.device) for k, v in inputs.items()}  # type: ignore[attr-defined]

    with torch.no_grad():  # type: ignore[attr-defined]
        outputs = cached.model(**inputs)  # type: ignore[attr-defined]
    logits = outputs.logits_per_image  # (1, num_labels)
    probs = logits.softmax(dim=-1).squeeze(0).cpu().tolist()
    pairs = list(zip(labels, probs, strict=True))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs


def is_loaded(model_name: str = DEFAULT_MODEL) -> bool:
    """Test helper — True if the named model has been loaded into the
    in-memory cache."""
    return model_name in _models


def clear_models() -> None:
    """Test hook — drops every cached model. Not exposed in the UI."""
    _models.clear()
