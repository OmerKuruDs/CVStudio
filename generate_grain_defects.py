"""Generate a YOLO-formatted dataset of synthetic grain-flattening defects.

Usage:
    python generate_grain_defects.py --input <folder_of_clean_metals> \\
                                     --output <dataset_folder>        \\
                                     --variants 5                     \\
                                     --intensity 0.6

Output layout (Ultralytics YOLO single-class format):
    <output>/
        images/  metal01_v0.png  metal01_v1.png  ...
        labels/  metal01_v0.txt  metal01_v1.txt  ...     (one line per defect)

Each label line: ``0 cx cy w h`` with values normalised to [0, 1].
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2

from cvstudio.defects import inject_grain_flattening


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="Folder of clean reference images.")
    parser.add_argument("--output", type=Path, required=True, help="Destination dataset folder.")
    parser.add_argument("--variants", type=int, default=3, help="How many defect variants to generate per input image.")
    parser.add_argument("--intensity", type=float, default=0.5, help="Defect severity in [0.0, 1.0].")
    parser.add_argument(
        "--directions", nargs="+", default=["horizontal", "vertical"],
        choices=["horizontal", "vertical"],
        help="Direction pool — each variant picks one at random.",
    )
    parser.add_argument("--feather", type=float, default=0.3, help="ROI edge feather fraction.")
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed for reproducibility.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    images_dir = args.output / "images"
    labels_dir = args.output / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(p for p in args.input.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})
    if not image_paths:
        print(f"No images found in {args.input}")
        return 1

    rng = random.Random(args.seed)
    total = 0
    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"Skipping unreadable: {image_path}")
            continue

        for variant_index in range(args.variants):
            variant_seed = args.seed * 10_000 + variant_index
            direction = rng.choice(args.directions)
            defective, label = inject_grain_flattening(
                image,
                flattening_direction=direction,
                intensity=args.intensity,
                feather_fraction=args.feather,
                seed=variant_seed,
            )
            stem = f"{image_path.stem}_v{variant_index}"
            cv2.imwrite(str(images_dir / f"{stem}{image_path.suffix}"), defective)
            cx, cy, w, h = label.yolo
            (labels_dir / f"{stem}.txt").write_text(
                f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n",
                encoding="utf-8",
            )
            total += 1
            print(f"  → {stem}  dir={direction}  bbox={label.bbox_xywh}")

    print(f"\nDone. {total} defective images + labels written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
