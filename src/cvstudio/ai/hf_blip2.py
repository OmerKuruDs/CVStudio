"""HuggingFace BLIP-2 backend for image captioning.

BLIP-2 ingests an image (no prompt required) and produces a free-form
caption — useful as a "what's in this picture" overview. We expose just
`caption(image, *, model_name, max_new_tokens) -> str`; setup mirrors
`hf_clip` (lazy import, module-level model cache, friendly error when
the `[ai]` extras aren't installed).

Default model is `Salesforce/blip2-opt-2.7b`, which is the smallest
BLIP-2 checkpoint — about 7 GB on disk after first download. The user
can swap it via the param panel for any HF caption-capable checkpoint
that the `Blip2ForConditionalGeneration` class understands.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from cvstudio.ai.hf_clip import HFExtrasMissing

DEFAULT_MODEL = "Salesforce/blip2-opt-2.7b"


@dataclass(frozen=True)
class _CachedModel:
    model: object  # transformers.Blip2ForConditionalGeneration
    processor: object  # transformers.Blip2Processor
    device: str


_models: dict[str, _CachedModel] = {}


def _ensure_imports() -> tuple[object, object, object]:
    try:
        import torch
        from transformers import Blip2ForConditionalGeneration, Blip2Processor
    except ImportError as exc:
        raise HFExtrasMissing(
            "HuggingFace extras not installed. "
            "Run: pip install -e .[ai]  (adds transformers + torch + pillow)"
        ) from exc
    return torch, Blip2ForConditionalGeneration, Blip2Processor


def _load_model(model_name: str) -> _CachedModel:
    if model_name in _models:
        return _models[model_name]
    torch, Blip2ForConditionalGeneration, Blip2Processor = _ensure_imports()  # type: ignore[misc]
    device = "cuda" if torch.cuda.is_available() else "cpu"  # type: ignore[attr-defined]
    # BLIP-2 weights are big; half-precision on GPU keeps the model in
    # consumer-card VRAM. CPU has to stay fp32 (some kernels don't
    # implement fp16 forward).
    dtype = torch.float16 if device == "cuda" else torch.float32  # type: ignore[attr-defined]
    model = Blip2ForConditionalGeneration.from_pretrained(  # type: ignore[attr-defined]
        model_name, torch_dtype=dtype
    ).to(device)
    model.eval()  # type: ignore[attr-defined]
    processor = Blip2Processor.from_pretrained(model_name)  # type: ignore[attr-defined]
    cached = _CachedModel(model=model, processor=processor, device=device)
    _models[model_name] = cached
    return cached


def _to_rgb_array(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def caption(
    image: np.ndarray,
    *,
    model_name: str = DEFAULT_MODEL,
    max_new_tokens: int = 30,
) -> str:
    """Generate a free-form caption for `image`. Empty results land as
    "(no caption produced)" so the caller never has to guard against
    an empty string."""
    torch, _, _ = _ensure_imports()
    cached = _load_model(model_name)

    rgb = _to_rgb_array(image)
    inputs = cached.processor(images=rgb, return_tensors="pt")  # type: ignore[attr-defined]
    # Mirror the model's dtype on the pixel tensor — mixed precision will
    # fail loudly otherwise.
    pixel_values = inputs["pixel_values"].to(cached.device)  # type: ignore[attr-defined]
    model_dtype = next(cached.model.parameters()).dtype  # type: ignore[attr-defined]
    pixel_values = pixel_values.to(model_dtype)

    with torch.no_grad():  # type: ignore[attr-defined]
        out_ids = cached.model.generate(  # type: ignore[attr-defined]
            pixel_values=pixel_values,
            max_new_tokens=int(max_new_tokens),
        )
    text = cached.processor.batch_decode(  # type: ignore[attr-defined]
        out_ids, skip_special_tokens=True
    )[0].strip()
    if not text:
        return "(no caption produced)"
    return text


def is_loaded(model_name: str = DEFAULT_MODEL) -> bool:
    """Test helper — True if the named model has been loaded into cache."""
    return model_name in _models


def clear_models() -> None:
    """Test hook — drops every cached model. Not exposed in the UI."""
    _models.clear()
