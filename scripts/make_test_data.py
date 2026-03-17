#!/usr/bin/env python3
"""scripts/make_test_data.py — Generate synthetic raw_data/ for pipeline testing.

Creates 2 fake cases, each with 4 images + 4 LabelMe JSON files that mimic
the Olympus GIF-H290Z endoscope format:
  * 1276×1020 JPEG
  * Left ~35 % of columns are black (metadata overlay region)
  * Circular-ish bright FOV on the right ~65 %
  * A bounding-box annotation placed inside the FOV

Usage::
    python scripts/make_test_data.py
    python scripts/make_test_data.py --out raw_data --cases 3
"""

import argparse
import json
import os
import random
from pathlib import Path

import cv2
import numpy as np

# ---- Image dimensions matching CLAUDE.md spec ----
IMG_W = 1276
IMG_H = 1020
META_FRAC = 0.37          # fraction of width occupied by metadata (black)
NOISE_SEED = 42


def _make_endoscope_image(
    img_w: int = IMG_W,
    img_h: int = IMG_H,
    meta_frac: float = META_FRAC,
    rng: np.random.Generator = None,
    nbi: bool = False,
) -> np.ndarray:
    """Synthesise a fake endoscope image (BGR uint8).

    * Left ``meta_frac`` columns → solid black (metadata area)
    * Remainder → circular FOV filled with pink/tan mucosa texture
    * Small darker ellipse simulating a flat lesion inside the FOV

    Args:
        img_w:      Image width in pixels.
        img_h:      Image height in pixels.
        meta_frac:  Fraction of width that is the black metadata area.
        rng:        NumPy random Generator for reproducibility.
        nbi:        If True, use a greenish NBI colour palette.

    Returns:
        HxWx3 uint8 BGR image.
    """
    if rng is None:
        rng = np.random.default_rng(NOISE_SEED)

    img = np.zeros((img_h, img_w, 3), dtype=np.uint8)

    meta_w = int(img_w * meta_frac)   # left black region width
    fov_x0 = meta_w
    fov_cx = (fov_x0 + img_w) // 2
    fov_cy = img_h // 2
    fov_rx = (img_w - fov_x0) // 2 - 10   # horizontal radius
    fov_ry = img_h // 2 - 10              # vertical radius

    # Draw circular FOV with mucosal texture
    # Base colour: pinkish WLI or greenish NBI
    base_r = (160, 100, 80) if not nbi else (60, 140, 60)   # (R, G, B)
    base_bgr = (base_r[2], base_r[1], base_r[0])

    yy, xx = np.mgrid[0:img_h, fov_x0:img_w]
    inside_fov = ((xx - fov_cx) ** 2 / fov_rx ** 2 +
                  (yy - fov_cy) ** 2 / fov_ry ** 2) <= 1.0

    for c, base_val in enumerate(base_bgr):
        chan = np.zeros((img_h, img_w - fov_x0), dtype=np.float32)
        chan[inside_fov] = base_val
        # Add gaussian noise for texture
        noise = rng.normal(0, 12, chan.shape).astype(np.float32)
        chan = np.clip(chan + noise * inside_fov, 0, 255).astype(np.uint8)
        img[:, fov_x0:, c] = chan

    # Draw a slightly darker flat lesion ellipse (simulating EGC 0-IIb)
    lesion_cx = fov_cx + rng.integers(-fov_rx // 4, fov_rx // 4)
    lesion_cy = fov_cy + rng.integers(-fov_ry // 4, fov_ry // 4)
    lesion_rx = rng.integers(30, 80)
    lesion_ry = rng.integers(20, 60)

    yyl, xxl = np.mgrid[0:img_h, fov_x0:img_w]
    inside_lesion = ((xxl - lesion_cx) ** 2 / lesion_rx ** 2 +
                     (yyl - lesion_cy) ** 2 / lesion_ry ** 2) <= 1.0
    inside_lesion &= inside_fov   # lesion is inside FOV

    darken = 40
    for c in range(3):
        chan = img[:, fov_x0:, c].astype(np.int16)
        chan[inside_lesion] -= darken
        img[:, fov_x0:, c] = np.clip(chan, 0, 255).astype(np.uint8)

    # Return also the lesion bounding box in original image coords
    ys, xs = np.where(inside_lesion)
    if len(xs):
        bbox = (
            int(xs.min()) + fov_x0,
            int(ys.min()),
            int(xs.max()) + fov_x0,
            int(ys.max()),
        )
    else:
        # Fallback bbox
        bbox = (
            lesion_cx - lesion_rx + fov_x0,
            lesion_cy - lesion_ry,
            lesion_cx + lesion_rx + fov_x0,
            lesion_cy + lesion_ry,
        )

    return img, bbox


def _labelme_json(
    image_filename: str,
    bbox: tuple,
    label: str,
    img_w: int = IMG_W,
    img_h: int = IMG_H,
) -> dict:
    """Build a minimal LabelMe JSON dict for one rectangle annotation.

    Args:
        image_filename: Relative image filename referenced inside the JSON.
        bbox:           ``(x1, y1, x2, y2)`` bounding box in pixel coords.
        label:          Shape label string (Chinese, e.g. '白光近景').
        img_w:          Image width.
        img_h:          Image height.

    Returns:
        Dict ready to serialise as JSON.
    """
    x1, y1, x2, y2 = bbox
    return {
        "version": "0.4.29",
        "flags": {},
        "shapes": [{
            "label": label,
            "points": [[float(x1), float(y1)], [float(x2), float(y2)]],
            "group_id": None,
            "shape_type": "rectangle",
            "flags": {}
        }],
        "imagePath": image_filename,
        "imageData": None,
        "imageHeight": img_h,
        "imageWidth": img_w,
    }


# Map (modality, view) → (Chinese label, image file suffix, use_nbi)
_SPEC = {
    ('WLI', 'distant'):  ('白光远景', '014', False),
    ('WLI', 'closeup'):  ('白光近景', '023', False),
    ('NBI', 'distant'):  ('NBI远景',  '026', True),
    ('NBI', 'closeup'):  ('NBI近景',  '029', True),
}


def make_case(case_dir: Path, seed: int = 0) -> None:
    """Generate one fake case directory with 4 images + 4 JSON files.

    Args:
        case_dir: Directory to create (created if absent).
        seed:     RNG seed for reproducibility.
    """
    case_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    for (modality, view), (label, file_stem, is_nbi) in _SPEC.items():
        img, bbox = _make_endoscope_image(rng=rng, nbi=is_nbi)

        img_filename = f"{file_stem}.jpg"
        img_path = case_dir / img_filename

        # Save JPEG
        cv2.imwrite(str(img_path), img)

        # Build JSON filename with Chinese label (matches real data convention)
        json_filename = f"{file_stem}{label}.json"
        json_data = _labelme_json(img_filename, bbox, label)

        with open(case_dir / json_filename, 'w', encoding='utf-8') as fh:
            json.dump(json_data, fh, ensure_ascii=False, indent=2)

    print(f"  Created {case_dir.name}/ with {len(_SPEC) * 2} files")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Generate synthetic raw_data/ for preprocess.py testing')
    parser.add_argument('--out',   default='raw_data',
                        help='Output root directory')
    parser.add_argument('--cases', type=int, default=2,
                        help='Number of fake cases to create')
    args = parser.parse_args()

    out_root = Path(args.out)
    print(f"Generating {args.cases} synthetic case(s) → {out_root.resolve()}")

    for i in range(1, args.cases + 1):
        case_id = f"case{i:03d}"
        make_case(out_root / case_id, seed=i)

    print(f"Done.  {args.cases} case(s) written to '{out_root}'.")


if __name__ == '__main__':
    main()
