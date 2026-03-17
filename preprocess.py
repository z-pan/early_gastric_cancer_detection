#!/usr/bin/env python3
"""preprocess.py — DSCMFNet raw-data preprocessing pipeline.

Converts raw Olympus GIF-H290Z endoscopic images + LabelMe JSON annotations
into cropped, normalised training data organised by modality and view.

Usage::

    python preprocess.py --raw_root raw_data/ --out_root processed_data/
    python preprocess.py --raw_root raw_data/ --out_root processed_data/ --view close-up
    python preprocess.py --raw_root raw_data/ --out_root processed_data/ --cases case001 case002
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw


# --------------------------------------------------------------------------- #
# 1.  FOV detection                                                            #
# --------------------------------------------------------------------------- #

def _moving_average(arr: np.ndarray, k: int) -> np.ndarray:
    """1-D moving average with edge-padding; output has the same length as input.

    Uses the cumsum trick: prepend 0 so that ``cs[i+k] - cs[i]`` gives the
    sum of ``k`` consecutive elements starting at index ``i``.

    Args:
        arr: 1-D float array of length n.
        k:   Kernel size (should be odd for symmetric padding).

    Returns:
        Smoothed array with the same length n as ``arr``.
    """
    pad = np.pad(arr, (k // 2, k - 1 - k // 2), mode='edge')  # length n + k - 1
    cs = np.concatenate([[0.0], np.cumsum(pad, dtype=float)])  # length n + k
    return (cs[k:] - cs[:-k]) / k


def detect_fov(
    img: np.ndarray,
    threshold: int = 30,
    smooth_k: int = 15,
    bright_frac: float = 0.5,
) -> Tuple[int, int, int, int]:
    """Detect the circular FOV bounding box of an Olympus endoscope image.

    Algorithm (per CLAUDE.md §Data Description):
      * Convert to greyscale and threshold at `threshold`.
      * For each *column*, compute the fraction of bright pixels in the
        middle 70 % of rows (15–85 %).  Smooth with a moving average.
        FOV left edge = first column whose smoothed brightness > ``bright_frac``.
        FOV right edge = last such column.
      * Same logic along rows to find top/bottom edges.

    Args:
        img:         HxWx3 uint8 BGR image.
        threshold:   Pixel brightness to call 'bright'.
        smooth_k:    Moving-average kernel size.
        bright_frac: Threshold on smoothed fraction to enter the FOV.

    Returns:
        ``(x0, y0, x1, y1)`` — inclusive crop coordinates in the original image.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    bright = (gray > threshold).astype(np.float32)  # H × W

    mid_r0, mid_r1 = int(H * 0.15), int(H * 0.85)
    mid_c0, mid_c1 = int(W * 0.15), int(W * 0.85)

    # ---- Left / Right -------------------------------------------------------
    col_frac = bright[mid_r0:mid_r1, :].mean(axis=0)          # shape (W,)
    col_smooth = _moving_average(col_frac, smooth_k)

    x0 = 0
    for c in range(W):
        if col_smooth[c] > bright_frac:
            x0 = c
            break

    x1 = W - 1
    for c in range(W - 1, -1, -1):
        if col_smooth[c] > bright_frac:
            x1 = c
            break

    # ---- Top / Bottom -------------------------------------------------------
    row_frac = bright[:, mid_c0:mid_c1].mean(axis=1)          # shape (H,)
    row_smooth = _moving_average(row_frac, smooth_k)

    y0 = 0
    for r in range(H):
        if row_smooth[r] > bright_frac:
            y0 = r
            break

    y1 = H - 1
    for r in range(H - 1, -1, -1):
        if row_smooth[r] > bright_frac:
            y1 = r
            break

    # Clamp to image bounds
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(W - 1, x1)
    y1 = min(H - 1, y1)
    return x0, y0, x1, y1


# --------------------------------------------------------------------------- #
# 2.  Label classification                                                     #
# --------------------------------------------------------------------------- #

def classify_label(label: str) -> Tuple[Optional[str], Optional[str]]:
    """Classify a LabelMe label string into (modality, view).

    Naming convention (CLAUDE.md §Label naming convention):
      * "NBI" / "nbi"  → NBI modality;  "白光" → WLI modality
      * "近景"          → close-up view; "远景" → distant view

    Args:
        label: Label string from the JSON or filename.

    Returns:
        ``(modality, view)`` where modality ∈ {``'WLI'``, ``'NBI'``} and
        view ∈ {``'closeup'``, ``'distant'``}, or ``None`` for each unknown field.
    """
    ll = label.lower()
    modality: Optional[str] = None
    view: Optional[str] = None

    if 'nbi' in ll:
        modality = 'NBI'
    elif '白光' in label:
        modality = 'WLI'

    if '近景' in label:
        view = 'closeup'
    elif '远景' in label:
        view = 'distant'

    return modality, view


# --------------------------------------------------------------------------- #
# 3.  LabelMe JSON parsing                                                     #
# --------------------------------------------------------------------------- #

def parse_labelme_json(json_path: Path) -> dict:
    """Load a LabelMe JSON file.

    Args:
        json_path: Path to the ``.json`` annotation file.

    Returns:
        Raw JSON dict with keys ``'shapes'``, ``'imagePath'``,
        ``'imageHeight'``, ``'imageWidth'``, etc.
    """
    with open(json_path, 'r', encoding='utf-8') as fh:
        return json.load(fh)


def shapes_to_mask(shapes: List[dict], height: int, width: int) -> np.ndarray:
    """Render LabelMe shapes (rectangle **or** polygon) into a binary mask.

    Args:
        shapes: List of shape dicts from a LabelMe JSON (already
                coordinate-transformed to crop space).
        height: Crop height in pixels.
        width:  Crop width in pixels.

    Returns:
        uint8 binary mask (values 0 / 255), shape ``(height, width)``.
    """
    mask_img = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask_img)

    for shape in shapes:
        pts = shape['points']
        stype = shape.get('shape_type', 'polygon')
        if stype == 'rectangle':
            x1, y1 = pts[0]
            x2, y2 = pts[1]
            x_min, x_max = sorted([x1, x2])
            y_min, y_max = sorted([y1, y2])
            draw.rectangle([x_min, y_min, x_max, y_max], fill=255)
        else:
            flat = [(float(p[0]), float(p[1])) for p in pts]
            if len(flat) >= 3:
                draw.polygon(flat, fill=255)

    return np.array(mask_img)


def transform_shapes(shapes: List[dict], x0: int, y0: int) -> List[dict]:
    """Subtract FOV origin from all shape point coordinates.

    Args:
        shapes: Original LabelMe shape dicts.
        x0: FOV left edge in original image pixel coords.
        y0: FOV top edge in original image pixel coords.

    Returns:
        New list of shape dicts with translated ``'points'``.
    """
    new_shapes = []
    for shape in shapes:
        s = dict(shape)
        s['points'] = [[p[0] - x0, p[1] - y0] for p in shape['points']]
        new_shapes.append(s)
    return new_shapes


def clamp_shapes(shapes: List[dict], crop_w: int, crop_h: int) -> List[dict]:
    """Clamp shape coordinates to stay within the crop dimensions.

    Args:
        shapes: Coordinate-transformed shape dicts.
        crop_w: Crop width.
        crop_h: Crop height.

    Returns:
        New list with all points clamped to ``[0, crop_w-1] × [0, crop_h-1]``.
    """
    new_shapes = []
    for shape in shapes:
        s = dict(shape)
        s['points'] = [
            [float(np.clip(p[0], 0, crop_w - 1)),
             float(np.clip(p[1], 0, crop_h - 1))]
            for p in shape['points']
        ]
        new_shapes.append(s)
    return new_shapes


# --------------------------------------------------------------------------- #
# 4.  Case-level orchestration                                                 #
# --------------------------------------------------------------------------- #

_IMG_EXTS = ['.jpg', '.jpeg', '.png', '.bmp']


def resolve_image_path(json_path: Path, img_path_str: str) -> Optional[Path]:
    """Resolve the image file referenced inside a LabelMe JSON.

    Tries the literal path first; if not found, probes common image extensions
    in the same directory as the JSON.

    Args:
        json_path:     Path to the JSON file (used to determine search dir).
        img_path_str:  Raw ``imagePath`` value from the JSON.

    Returns:
        Resolved ``Path`` if found, else ``None``.
    """
    if not img_path_str:
        return None

    candidate = json_path.parent / img_path_str
    if candidate.exists():
        return candidate

    stem = Path(img_path_str).stem
    for ext in _IMG_EXTS:
        p = json_path.parent / (stem + ext)
        if p.exists():
            return p
    return None


def classify_json(json_path: Path) -> Tuple[Optional[str], Optional[str], Optional[Path]]:
    """Determine modality, view, and image path from a LabelMe JSON.

    Tries labels inside ``'shapes'`` first; falls back to classifying the
    JSON filename itself.

    Args:
        json_path: Path to the ``.json`` file.

    Returns:
        ``(modality, view, image_path)`` — any element may be ``None``.
    """
    data = parse_labelme_json(json_path)
    shapes = data.get('shapes', [])

    modality: Optional[str] = None
    view: Optional[str] = None

    for shape in shapes:
        m, v = classify_label(shape.get('label', ''))
        if m:
            modality = m
        if v:
            view = v
        if modality and view:
            break

    # Fallback: classify from JSON filename
    if not (modality and view):
        m, v = classify_label(json_path.stem)
        modality = modality or m
        view = view or v

    img_path = resolve_image_path(json_path, data.get('imagePath', ''))
    return modality, view, img_path


def process_case(
    case_dir: Path,
    out_root: Path,
    view_filter: str,
    case_id: str,
) -> List[dict]:
    """Process all annotations inside one case directory.

    Args:
        case_dir:     Path to the raw case folder (e.g. ``raw_data/case001``).
        out_root:     Root of ``processed_data/``.
        view_filter:  ``'close-up'``, ``'distant'``, or ``'all'``.
        case_id:      Case identifier string (e.g. ``'case001'``).

    Returns:
        List of result dicts with processing metadata for each output file.
    """
    results: List[dict] = []
    json_files = sorted(case_dir.glob('*.json'))

    if not json_files:
        print(f"  [WARN] No JSON files in {case_dir}")
        return results

    for json_path in json_files:
        modality, view, img_path = classify_json(json_path)

        if modality is None or view is None:
            print(f"  [SKIP] {json_path.name}: "
                  f"cannot classify (modality={modality}, view={view})")
            continue

        # Apply view filter
        if view_filter == 'close-up' and view != 'closeup':
            continue
        if view_filter == 'distant' and view != 'distant':
            continue

        if img_path is None or not img_path.exists():
            print(f"  [SKIP] {json_path.name}: image not found")
            continue

        # --- Load image -------------------------------------------------------
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"  [SKIP] {img_path.name}: cv2.imread failed")
            continue

        H, W = img_bgr.shape[:2]

        # --- FOV detection ----------------------------------------------------
        x0, y0, x1, y1 = detect_fov(img_bgr)
        crop_w = x1 - x0 + 1
        crop_h = y1 - y0 + 1

        # Guard against degenerate crops
        if crop_w < 16 or crop_h < 16:
            print(f"  [WARN] {json_path.name}: degenerate FOV crop "
                  f"({crop_w}×{crop_h}), falling back to full image")
            x0, y0, x1, y1 = 0, 0, W - 1, H - 1
            crop_w, crop_h = W, H

        # --- Crop image -------------------------------------------------------
        img_crop = img_bgr[y0:y1 + 1, x0:x1 + 1]

        # --- Transform + clamp annotation shapes ------------------------------
        data = parse_labelme_json(json_path)
        shapes = data.get('shapes', [])
        shapes_xf = transform_shapes(shapes, x0, y0)
        shapes_xf = clamp_shapes(shapes_xf, crop_w, crop_h)

        # --- Generate binary mask (0/255) ------------------------------------
        mask = shapes_to_mask(shapes_xf, crop_h, crop_w)

        # --- Build output filename -------------------------------------------
        out_name = f"{case_id}_{modality}_{view}.png"

        # --- Save cropped image ----------------------------------------------
        img_out_dir = out_root / modality
        img_out_dir.mkdir(parents=True, exist_ok=True)
        img_out_path = img_out_dir / out_name
        img_rgb = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
        Image.fromarray(img_rgb).save(img_out_path)

        # --- Save binary mask ------------------------------------------------
        mask_out_dir = out_root / f"Mask_{modality}"
        mask_out_dir.mkdir(parents=True, exist_ok=True)
        mask_out_path = mask_out_dir / out_name
        Image.fromarray(mask).save(mask_out_path)

        # --- Save polygon JSON for MedSAM review -----------------------------
        poly_dir = out_root / 'polygon_annotations'
        poly_dir.mkdir(parents=True, exist_ok=True)
        poly_json = {
            'version': data.get('version', '0.4.29'),
            'shapes': shapes_xf,
            'imagePath': out_name,
            'imageHeight': crop_h,
            'imageWidth': crop_w,
        }
        with open(poly_dir / f"{case_id}_{modality}_{view}.json",
                  'w', encoding='utf-8') as fh:
            json.dump(poly_json, fh, ensure_ascii=False, indent=2)

        result = {
            'case_id':         case_id,
            'modality':        modality,
            'view':            view,
            'src_image':       str(img_path),
            'fov_box':         (x0, y0, x1, y1),
            'orig_size':       (W, H),
            'crop_size':       (crop_w, crop_h),
            'mask_positive_px': int(mask.sum() // 255),
            'out_image':       str(img_out_path),
            'out_mask':        str(mask_out_path),
        }
        results.append(result)

        print(f"  [OK] {case_id} {modality}/{view}  "
              f"FOV=({x0},{y0})-({x1},{y1})  "
              f"crop={crop_w}×{crop_h}  "
              f"mask_px={result['mask_positive_px']}")

    return results


# --------------------------------------------------------------------------- #
# 5.  Entry point                                                              #
# --------------------------------------------------------------------------- #

def main() -> None:
    """CLI entry point for the preprocessing pipeline."""
    parser = argparse.ArgumentParser(
        description='DSCMFNet preprocessing pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--raw_root',  default='raw_data',
                        help='Root directory of raw case folders')
    parser.add_argument('--out_root',  default='processed_data',
                        help='Output directory for processed data')
    parser.add_argument('--view',      default='all',
                        choices=['close-up', 'distant', 'all'],
                        help='Which views to process')
    parser.add_argument('--cases', nargs='*', default=None,
                        help='Specific case IDs to process (default: all)')
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)

    if not raw_root.exists():
        print(f"ERROR: --raw_root '{raw_root}' does not exist.", file=sys.stderr)
        sys.exit(1)

    out_root.mkdir(parents=True, exist_ok=True)

    # Discover case directories
    case_dirs = sorted(d for d in raw_root.iterdir() if d.is_dir())
    if args.cases:
        case_dirs = [d for d in case_dirs if d.name in args.cases]

    if not case_dirs:
        print("No case directories found.")
        sys.exit(0)

    print(f"Found {len(case_dirs)} case(s).  view_filter='{args.view}'")
    print(f"Output → {out_root.resolve()}\n")

    all_results: List[dict] = []
    for case_dir in case_dirs:
        print(f"Processing {case_dir.name} …")
        results = process_case(case_dir, out_root, args.view, case_dir.name)
        all_results.extend(results)

    # ---- Summary -----------------------------------------------------------
    total = len(all_results)
    wli_n    = sum(1 for r in all_results if r['modality'] == 'WLI')
    nbi_n    = sum(1 for r in all_results if r['modality'] == 'NBI')
    cu_n     = sum(1 for r in all_results if r['view'] == 'closeup')
    dist_n   = sum(1 for r in all_results if r['view'] == 'distant')
    avg_mask = (sum(r['mask_positive_px'] for r in all_results) / total
                if total else 0)

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total processed : {total}")
    print(f"  WLI           : {wli_n}")
    print(f"  NBI           : {nbi_n}")
    print(f"  close-up      : {cu_n}")
    print(f"  distant       : {dist_n}")
    print(f"  Avg mask px   : {avg_mask:.0f}")
    print("=" * 60)

    if total:
        print("\nOutput files:")
        for r in all_results:
            print(f"  {r['out_image']}")
            print(f"  {r['out_mask']}")


if __name__ == '__main__':
    main()
