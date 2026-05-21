"""Synthetic defect injection for industrial-CV data augmentation.

These functions take a clean reference image and return a defective copy
plus YOLO-ready label metadata, so a small seed dataset can be expanded
into a labelled training set without manual annotation. Unlike the
``cvstudio.operations`` modules these are pure library functions — no
``OperationSpec`` wrapping — because the (image, label) return shape does
not fit the single-ndarray pipeline contract.
"""

from cvstudio.defects.grain_flattening import (
    GrainFlatteningLabel,
    inject_grain_flattening,
)

__all__ = ["GrainFlatteningLabel", "inject_grain_flattening"]
