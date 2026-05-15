"""Minimal Ollama HTTP client for vision-language queries.

Speaks the `POST /api/generate` endpoint with a single image (base64 JPEG) and
a text prompt. We intentionally avoid `requests` so the package stays
dependency-free for users who do not opt into the AI stack.

Streaming responses are disabled (`stream=False`) so the worker thread can
treat a generation as a single blocking call. The pipeline's worker already
runs on its own QThread, so blocking there is acceptable and keeps this
module synchronous.
"""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass

import cv2
import numpy as np

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "llava"
DEFAULT_TIMEOUT_SECONDS = 120.0


class OllamaError(RuntimeError):
    """Raised when the Ollama backend is unreachable or returns an error.

    We surface this as a separate type so callers (the VLM operation) can
    catch it and render a banner without swallowing unrelated runtime
    exceptions from the pipeline worker."""


@dataclass(frozen=True)
class OllamaResponse:
    text: str
    model: str


def encode_image_jpeg(image: np.ndarray, quality: int = 85) -> bytes:
    """Encode `image` (BGR or grayscale ndarray) to JPEG bytes. Grayscale is
    promoted to BGR first because llava-style models expect RGB-like input."""
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise OllamaError("Failed to encode image as JPEG")
    return bytes(buf)


def generate(
    prompt: str,
    image: np.ndarray,
    *,
    model: str = DEFAULT_MODEL,
    host: str = DEFAULT_HOST,
    temperature: float = 0.2,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    opener: urllib.request.OpenerDirector | None = None,
) -> OllamaResponse:
    """Send `image` + `prompt` to Ollama and return the model's reply.

    `opener` is injected for tests — production code uses the module default
    (`urllib.request.urlopen`), tests supply a fake that returns a canned JSON
    payload without touching the network.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [base64.b64encode(encode_image_jpeg(image)).decode("ascii")],
        "stream": False,
        "options": {"temperature": float(temperature)},
    }
    body = json.dumps(payload).encode("utf-8")
    url = host.rstrip("/") + "/api/generate"
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        if opener is not None:
            response = opener.open(request, timeout=timeout)
        else:
            response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.URLError as exc:
        raise OllamaError(f"Could not reach Ollama at {host}: {exc.reason}") from exc
    except TimeoutError as exc:
        raise OllamaError(f"Ollama request timed out after {timeout:.0f}s") from exc

    with response:
        raw = response.read()
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OllamaError(f"Ollama returned malformed JSON: {exc}") from exc

    text = str(data.get("response", "")).strip()
    if not text:
        # Ollama returns 200 with an empty "response" when the model errors
        # mid-stream; the real cause is usually in data["error"].
        err = data.get("error")
        if err:
            raise OllamaError(f"Ollama error: {err}")
        raise OllamaError("Ollama returned an empty response")
    return OllamaResponse(text=text, model=str(data.get("model", model)))


def stream_generate(
    prompt: str,
    image: np.ndarray,
    *,
    model: str = DEFAULT_MODEL,
    host: str = DEFAULT_HOST,
    temperature: float = 0.2,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    opener: urllib.request.OpenerDirector | None = None,
) -> Iterator[str]:
    """Stream Ollama's reply token-by-token. Yields each `response` chunk as
    it arrives over the line-delimited JSON stream.

    Connection failures raise `OllamaError` before the first yield; backend
    errors that surface mid-stream raise `OllamaError` from the iterator and
    abort early — partial output is whatever the caller accumulated before
    the raise.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [base64.b64encode(encode_image_jpeg(image)).decode("ascii")],
        "stream": True,
        "options": {"temperature": float(temperature)},
    }
    body = json.dumps(payload).encode("utf-8")
    url = host.rstrip("/") + "/api/generate"
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        if opener is not None:
            response = opener.open(request, timeout=timeout)
        else:
            response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.URLError as exc:
        raise OllamaError(f"Could not reach Ollama at {host}: {exc.reason}") from exc
    except TimeoutError as exc:
        raise OllamaError(f"Ollama request timed out after {timeout:.0f}s") from exc

    with response:
        for raw_line in response:
            line = raw_line.strip()
            if not line:
                continue
            try:
                data = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                # Half-flushed line — wait for the next iteration.
                continue
            err = data.get("error")
            if err:
                raise OllamaError(f"Ollama error: {err}")
            token = data.get("response", "")
            if token:
                yield token
            if data.get("done"):
                return
