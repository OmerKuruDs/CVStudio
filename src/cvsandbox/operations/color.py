"""Color-space and channel operations.

These ops generally accept 3-channel BGR input. `to_grayscale` and `channel`
gracefully pass through if the image is already single-channel; `to_hsv` is
explicit about needing 3 channels.

`invert` is here rather than in filtering because it's the simplest color
transformation we have.
"""

from __future__ import annotations

import cv2
import numpy as np

from cvsandbox.core.operation import OperationSpec, Parameter


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _to_hsv(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("HSV conversion requires a 3-channel BGR image")
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def _invert(image: np.ndarray) -> np.ndarray:
    return cv2.bitwise_not(image)


def _channel(image: np.ndarray, channel: int) -> np.ndarray:
    if image.ndim == 2:
        return image  # nothing to extract
    idx = int(channel) % image.shape[2]
    return image[:, :, idx]


TO_GRAYSCALE = OperationSpec(
    id="color.to_grayscale",
    name="To Grayscale",
    category="Color",
    description="Converts BGR to single-channel grayscale. Pass-through if already grayscale.",
    parameters=(),
    func=_to_grayscale,
)


TO_HSV = OperationSpec(
    id="color.to_hsv",
    name="To HSV",
    category="Color",
    description="Converts BGR to HSV. Useful before thresholding on hue/saturation.",
    parameters=(),
    func=_to_hsv,
)


INVERT = OperationSpec(
    id="color.invert",
    name="Invert",
    category="Color",
    description="255 - pixel. Works on any channel count.",
    parameters=(),
    func=_invert,
)


CHANNEL = OperationSpec(
    id="color.channel",
    name="Extract Channel",
    category="Color",
    description="Outputs a single channel by index (0/1/2 = B/G/R for BGR input, H/S/V for HSV).",
    parameters=(
        Parameter(
            name="channel",
            kind="int",
            default=0,
            min=0,
            max=2,
            label="Channel index",
        ),
    ),
    func=_channel,
)


ALL: tuple[OperationSpec, ...] = (TO_GRAYSCALE, TO_HSV, INVERT, CHANNEL)
