"""HuggingFace OWL-ViT backend for zero-shot object detection.

OWL-ViT takes an image plus a list of free-form text prompts and returns
bounding boxes for regions matching each prompt, along with a confidence
score per box. We expose the smallest possible surface —
`detect(image, prompts) -> [Detection, ...]` sorted by score descending —
and share the same lazy-import / friendly-error pattern as `hf_clip`.

Model + processor are cached at module level keyed by model name so the
multi-second model load only happens on the first call. Inference uses
CUDA when available, falls back to CPU otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from cvstudio.ai.hf_clip import HFExtrasMissing

DEFAULT_MODEL = "google/owlvit-base-patch32"


@dataclass(frozen=True)
class Detection:
    label: str
    score: float
    box: tuple[int, int, int, int]
    """(x1, y1, x2, y2) in pixel coordinates of the input image."""


@dataclass(frozen=True)
class _CachedModel:
    model: object  # transformers.OwlViTForObjectDetection — typed as object to avoid hard dep
    processor: object  # transformers.OwlViTProcessor
    device: str


_models: dict[str, _CachedModel] = {}


def _ensure_imports() -> tuple[object, object, object]:
    """Lazy import. Raises HFExtrasMissing with a fix-it message when the
    `ai` extras aren't installed."""
    try:
        import torch
        from transformers import OwlViTForObjectDetection, OwlViTProcessor
    except ImportError as exc:
        raise HFExtrasMissing(
            "HuggingFace extras not installed. "
            "Run: pip install -e .[ai]  (adds transformers + torch + pillow)"
        ) from exc
    return torch, OwlViTForObjectDetection, OwlViTProcessor


def _load_model(model_name: str) -> _CachedModel:
    if model_name in _models:
        return _models[model_name]
    torch, OwlViTForObjectDetection, OwlViTProcessor = _ensure_imports()  # type: ignore[misc]
    device = "cuda" if torch.cuda.is_available() else "cpu"  # type: ignore[attr-defined]
    model = OwlViTForObjectDetection.from_pretrained(model_name).to(device)  # type: ignore[attr-defined]
    model.eval()  # type: ignore[attr-defined]
    processor = OwlViTProcessor.from_pretrained(model_name)  # type: ignore[attr-defined]
    cached = _CachedModel(model=model, processor=processor, device=device)
    _models[model_name] = cached
    return cached


def _to_rgb_array(image: np.ndarray) -> np.ndarray:
    """OWL-ViT's processor wants H×W×3 RGB; OpenCV inputs are BGR / grayscale
    / BGRA, so coerce here."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def detect(
    image: np.ndarray,
    prompts: list[str],
    *,
    model_name: str = DEFAULT_MODEL,
    score_threshold: float = 0.1,
) -> list[Detection]:
    """Run zero-shot object detection. Returns `Detection`s sorted by
    score descending. Boxes are clipped to the image's pixel grid
    (`int` coords). Empty `prompts` returns `[]` without loading the
    model."""
    if not prompts:
        return []

    torch, _, _ = _ensure_imports()
    cached = _load_model(model_name)

    rgb = _to_rgb_array(image)
    # OWL-ViT's processor takes text as a list-of-lists: one inner list per
    # image. We always send a single image so the outer list has length 1.
    inputs = cached.processor(  # type: ignore[attr-defined]
        text=[prompts],
        images=rgb,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(cached.device) for k, v in inputs.items()}  # type: ignore[attr-defined]

    with torch.no_grad():  # type: ignore[attr-defined]
        outputs = cached.model(**inputs)  # type: ignore[attr-defined]

    h, w = image.shape[:2]
    target_sizes = torch.tensor([[h, w]])  # type: ignore[attr-defined]
    results = cached.processor.post_process_object_detection(  # type: ignore[attr-defined]
        outputs=outputs,
        target_sizes=target_sizes,
        threshold=float(score_threshold),
    )[0]

    detections: list[Detection] = []
    for box, score, label_idx in zip(
        results["boxes"], results["scores"], results["labels"], strict=True
    ):
        x1, y1, x2, y2 = box.cpu().tolist()
        label_str = prompts[int(label_idx)]
        # Clamp to int pixel grid + image bounds. Drop degenerate boxes that
        # post-clip end up zero-area (rare but possible at edges).
        ix1 = max(0, min(w, int(round(x1))))
        iy1 = max(0, min(h, int(round(y1))))
        ix2 = max(0, min(w, int(round(x2))))
        iy2 = max(0, min(h, int(round(y2))))
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        detections.append(
            Detection(
                label=label_str,
                score=float(score),
                box=(ix1, iy1, ix2, iy2),
            )
        )
    detections.sort(key=lambda d: d.score, reverse=True)
    return detections


def is_loaded(model_name: str = DEFAULT_MODEL) -> bool:
    """Test helper — True if the named model has been loaded into cache."""
    return model_name in _models


def clear_models() -> None:
    """Test hook — drops every cached model. Not exposed in the UI."""
    _models.clear()
