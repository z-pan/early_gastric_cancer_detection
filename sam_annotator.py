#!/usr/bin/env python3
"""sam_annotator.py — MedSAM-based bbox-to-mask refinement pipeline.

Upgrades bounding-box annotations produced by ``preprocess.py`` into
pixel-accurate segmentation masks using MedSAM (or vanilla SAM), then
exports three artefacts per image:

  1. Binary mask PNG     → ``processed_data/Mask_{MODALITY}/``
  2. LabelMe polygon JSON → ``processed_data/polygon_annotations/``
  3. Side-by-side review  → ``processed_data/reviews/``

Usage::

    # Full MedSAM run (single case)
    python sam_annotator.py \\
        --data_root processed_data \\
        --sam_checkpoint path/to/medsam_vit_b.pth \\
        --model_type vit_b \\
        --cases case001

    # Batch all cases
    python sam_annotator.py \\
        --data_root processed_data \\
        --sam_checkpoint path/to/medsam_vit_b.pth \\
        --batch

    # Pipeline test without SAM (pure bbox mask)
    python sam_annotator.py --data_root processed_data --no_sam --batch

Install MedSAM::

    pip install git+https://github.com/facebookresearch/segment-anything.git
    # Then download MedSAM checkpoint:
    # https://drive.google.com/drive/folders/1ETWmi4AiniJeWOt6HAsYgTjYv_fkgzoN
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

# --------------------------------------------------------------------------- #
# Optional SAM import — graceful degradation                                  #
# --------------------------------------------------------------------------- #

SAM_AVAILABLE = False
SamPredictor = None  # type: ignore[assignment]
sam_model_registry: Optional[dict] = None  # type: ignore[assignment]

try:
    from segment_anything import SamPredictor as _SP
    from segment_anything import sam_model_registry as _SMR

    SamPredictor = _SP
    sam_model_registry = _SMR
    SAM_AVAILABLE = True
except ImportError:
    pass


# --------------------------------------------------------------------------- #
# 1.  Annotation helpers                                                       #
# --------------------------------------------------------------------------- #

def load_annotation(json_path: Path) -> Tuple[Optional[np.ndarray], dict]:
    """Load a LabelMe JSON and extract the first shape's bounding box.

    Handles both ``rectangle`` (two corner points) and ``polygon`` (compute
    enclosing bbox) shape types.

    Args:
        json_path: Path to the LabelMe ``.json`` file.

    Returns:
        Tuple of ``(bbox, raw_json_dict)`` where ``bbox`` is an
        ``int32`` array ``[x1, y1, x2, y2]`` in image pixel coords, or
        ``None`` if no usable shape was found.
    """
    with open(json_path, 'r', encoding='utf-8') as fh:
        data = json.load(fh)

    shapes = data.get('shapes', [])
    if not shapes:
        return None, data

    shape = shapes[0]
    pts = np.array(shape['points'], dtype=np.float32)
    stype = shape.get('shape_type', 'polygon')

    if stype == 'rectangle' and len(pts) == 2:
        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)
    else:
        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)

    bbox = np.array([int(x_min), int(y_min), int(x_max), int(y_max)],
                    dtype=np.int32)
    return bbox, data


def bbox_to_mask(
    bbox: np.ndarray, height: int, width: int
) -> np.ndarray:
    """Rasterise a bounding box directly to a binary uint8 mask.

    Args:
        bbox:   ``[x1, y1, x2, y2]`` in pixel coords.
        height: Image height.
        width:  Image width.

    Returns:
        uint8 mask with 1 inside the box.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    x1, y1, x2, y2 = bbox
    x1 = int(np.clip(x1, 0, width - 1))
    y1 = int(np.clip(y1, 0, height - 1))
    x2 = int(np.clip(x2, 0, width - 1))
    y2 = int(np.clip(y2, 0, height - 1))
    mask[y1:y2 + 1, x1:x2 + 1] = 1
    return mask


# --------------------------------------------------------------------------- #
# 2.  MedSAM inference                                                        #
# --------------------------------------------------------------------------- #

def _score_candidate(
    mask: np.ndarray,
    iou_score: float,
    bbox: np.ndarray,
) -> float:
    """Score one SAM candidate mask.

    Formula (CLAUDE.md §sam_annotator):
        score = iou_score × 0.5
              + bbox_containment × 0.3
              − area_ratio_penalty
              − connectivity_penalty

    Args:
        mask:      Boolean / uint8 binary mask (H × W).
        iou_score: Predicted IoU from SAM.
        bbox:      ``[x1, y1, x2, y2]``.

    Returns:
        Scalar score (higher is better).
    """
    mask_bin = mask.astype(bool)
    mask_area = int(mask_bin.sum())

    x1, y1, x2, y2 = bbox.tolist()
    bbox_area = max(1, (x2 - x1) * (y2 - y1))

    # Bbox containment: fraction of mask pixels that lie inside the bbox
    h, w = mask_bin.shape
    roi = np.zeros((h, w), dtype=bool)
    roi[y1:y2 + 1, x1:x2 + 1] = True
    containment = (mask_bin & roi).sum() / max(1, mask_area)

    # Area-ratio penalty: punish masks that are ≥ 3× the bbox area
    area_ratio = mask_area / bbox_area
    area_penalty = max(0.0, area_ratio - 3.0) * 0.15

    # Connectivity penalty: prefer single connected component
    num_labels, _ = cv2.connectedComponents(
        mask_bin.astype(np.uint8), connectivity=8)
    n_components = max(0, num_labels - 1)
    connectivity_penalty = max(0, n_components - 1) * 0.1

    return iou_score * 0.5 + containment * 0.3 - area_penalty - connectivity_penalty


def run_medsam(
    predictor: "SamPredictor",  # type: ignore[name-defined]
    img_rgb: np.ndarray,
    bbox: np.ndarray,
) -> np.ndarray:
    """Run SAM/MedSAM with bbox + centre-point prompt and pick the best mask.

    Args:
        predictor: Initialised ``SamPredictor`` (SAM or MedSAM weights).
        img_rgb:   H × W × 3 uint8 RGB image.
        bbox:      ``[x1, y1, x2, y2]`` in pixel coords.

    Returns:
        Binary uint8 mask (values 0 / 1) of the same spatial size as ``img_rgb``.
    """
    predictor.set_image(img_rgb)

    x1, y1, x2, y2 = bbox.tolist()
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    point_coords = np.array([[cx, cy]], dtype=np.float32)
    point_labels = np.array([1], dtype=np.int32)    # 1 = foreground
    box_prompt = np.array([x1, y1, x2, y2], dtype=np.float32)

    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=box_prompt,
        multimask_output=True,
    )
    # masks: (3, H, W) bool;  scores: (3,) float

    candidate_scores = [
        _score_candidate(masks[i], float(scores[i]), bbox)
        for i in range(len(masks))
    ]
    best_idx = int(np.argmax(candidate_scores))
    return masks[best_idx].astype(np.uint8)


# --------------------------------------------------------------------------- #
# 3.  Post-processing                                                         #
# --------------------------------------------------------------------------- #

def postprocess_mask(mask: np.ndarray, min_area: int = 50) -> np.ndarray:
    """Clean up a raw SAM mask.

    Steps (CLAUDE.md):
      1. Remove connected components < ``min_area`` pixels.
      2. Fill enclosed holes.
      3. Morphological close (5×5 ellipse, 2 iterations).
      4. Morphological open  (5×5 ellipse, 1 iteration).

    Args:
        mask:     uint8 binary mask (values 0 / 1).
        min_area: Minimum component area to keep.

    Returns:
        Processed uint8 mask (values 0 / 1).
    """
    m = mask.astype(np.uint8)

    # --- Remove small components -------------------------------------------
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        m, connectivity=8)
    cleaned = np.zeros_like(m)
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 1

    # --- Fill enclosed holes -----------------------------------------------
    # Holes = background components that don't touch the image border.
    inv = 1 - cleaned
    n_inv, inv_labels, inv_stats, _ = cv2.connectedComponentsWithStats(
        inv, connectivity=8)
    h, w = cleaned.shape
    filled = cleaned.copy()
    for i in range(1, n_inv):
        comp = (inv_labels == i)
        touches = (comp[0, :].any() or comp[-1, :].any()
                   or comp[:, 0].any() or comp[:, -1].any())
        if not touches:
            filled[comp] = 1

    # --- Morphological smoothing -------------------------------------------
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    smoothed = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, kernel, iterations=2)
    smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN,  kernel, iterations=1)

    return smoothed


# --------------------------------------------------------------------------- #
# 4.  Mask → polygon                                                          #
# --------------------------------------------------------------------------- #

def mask_to_labelme_shapes(
    mask: np.ndarray,
    label: str,
    epsilon_factor: float = 0.005,
) -> List[dict]:
    """Convert a binary mask to a list of LabelMe polygon shapes.

    Uses ``cv2.findContours`` + ``cv2.approxPolyDP`` to simplify each
    external contour.

    Args:
        mask:           uint8 binary mask (values 0 / 1 or 0 / 255).
        label:          Shape label string (e.g. ``'白光近景'``).
        epsilon_factor: DP simplification factor relative to arc length.
                        Smaller → more vertices.

    Returns:
        List of LabelMe shape dicts with ``shape_type='polygon'``.
    """
    m = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    shapes = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 10:
            continue
        perimeter = cv2.arcLength(cnt, closed=True)
        epsilon = max(1.0, epsilon_factor * perimeter)
        approx = cv2.approxPolyDP(cnt, epsilon, closed=True)
        points = [[float(p[0][0]), float(p[0][1])] for p in approx]
        if len(points) < 3:
            continue
        shapes.append({
            'label':      label,
            'points':     points,
            'group_id':   None,
            'shape_type': 'polygon',
            'flags':      {},
        })

    return shapes


# --------------------------------------------------------------------------- #
# 5.  Review visualisation                                                    #
# --------------------------------------------------------------------------- #

_BBOX_COLOUR    = (0, 255, 0)    # green
_MASK_COLOUR    = (255, 80, 80)  # reddish overlay
_CONTOUR_COLOUR = (0, 220, 255)  # cyan
_MASK_ALPHA     = 0.35


def make_review_image(
    img_rgb: np.ndarray,
    bbox: np.ndarray,
    mask: np.ndarray,
    title_left: str = 'BBox prompt',
    title_right: str = 'SAM mask',
) -> np.ndarray:
    """Build a side-by-side review image (left: bbox, right: mask overlay).

    Args:
        img_rgb:     H × W × 3 uint8 RGB source image.
        bbox:        ``[x1, y1, x2, y2]`` bounding box.
        mask:        uint8 binary mask (values 0 / 1), same H × W.
        title_left:  Text label drawn on the left panel.
        title_right: Text label drawn on the right panel.

    Returns:
        Horizontally stacked H × 2W × 3 uint8 BGR image for ``cv2.imwrite``.
    """
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    H, W = img_bgr.shape[:2]

    # ---- Left panel: image + green bbox -----------------------------------
    left = img_bgr.copy()
    x1, y1, x2, y2 = bbox.tolist()
    cv2.rectangle(left, (x1, y1), (x2, y2), _BBOX_COLOUR, 2)

    # ---- Right panel: image + translucent mask + contour ------------------
    right = img_bgr.copy()
    if mask.sum() > 0:
        overlay = right.copy()
        overlay[mask > 0] = _MASK_COLOUR
        cv2.addWeighted(overlay, _MASK_ALPHA, right, 1 - _MASK_ALPHA, 0, right)

        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(right, contours, -1, _CONTOUR_COLOUR, 2)

    # ---- Titles -----------------------------------------------------------
    font = cv2.FONT_HERSHEY_SIMPLEX
    for panel, title in [(left, title_left), (right, title_right)]:
        cv2.putText(panel, title, (8, 24), font, 0.7, (255, 255, 255), 2,
                    cv2.LINE_AA)
        cv2.putText(panel, title, (8, 24), font, 0.7, (0, 0, 0), 1,
                    cv2.LINE_AA)

    return np.concatenate([left, right], axis=1)


# --------------------------------------------------------------------------- #
# 6.  Per-image processing                                                    #
# --------------------------------------------------------------------------- #

def process_image(
    img_path: Path,
    json_path: Path,
    out_mask_dir: Path,
    out_poly_dir: Path,
    out_review_dir: Path,
    predictor: Optional["SamPredictor"],  # type: ignore[name-defined]
    no_sam: bool,
) -> dict:
    """Process a single (image, annotation) pair.

    Args:
        img_path:       Path to the cropped PNG image.
        json_path:      Path to the corresponding polygon-annotations JSON.
        out_mask_dir:   Directory to save the binary mask PNG.
        out_poly_dir:   Directory to save the updated polygon JSON.
        out_review_dir: Directory to save the review JPEG.
        predictor:      Initialised ``SamPredictor``, or ``None``.
        no_sam:         If ``True``, skip SAM and use raw bbox mask.

    Returns:
        Dict with processing metadata.
    """
    stem = img_path.stem

    # ---- Load image -------------------------------------------------------
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"  [SKIP] Cannot read image: {img_path.name}")
        return {}
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = img_rgb.shape[:2]

    # ---- Load bbox annotation --------------------------------------------
    bbox, raw_data = load_annotation(json_path)
    if bbox is None:
        print(f"  [SKIP] No usable annotation in: {json_path.name}")
        return {}

    label = ''
    if raw_data.get('shapes'):
        label = raw_data['shapes'][0].get('label', '')

    # ---- Generate mask ----------------------------------------------------
    method = 'bbox'
    if not no_sam and predictor is not None:
        try:
            raw_mask = run_medsam(predictor, img_rgb, bbox)
            method = 'medsam'
        except Exception as exc:
            print(f"  [WARN] SAM failed ({exc}), falling back to bbox mask")
            raw_mask = bbox_to_mask(bbox, H, W)
    else:
        raw_mask = bbox_to_mask(bbox, H, W)

    # ---- Post-process -----------------------------------------------------
    mask = postprocess_mask(raw_mask)
    mask_pos_px = int(mask.sum())

    # ---- Save binary mask PNG (0/255) ------------------------------------
    out_mask_dir.mkdir(parents=True, exist_ok=True)
    mask_path = out_mask_dir / f"{stem}.png"
    Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)

    # ---- Build + save polygon JSON ---------------------------------------
    out_poly_dir.mkdir(parents=True, exist_ok=True)
    poly_shapes = mask_to_labelme_shapes(mask, label=label)
    # Fall back to bbox shape if mask is empty or contour failed
    if not poly_shapes:
        x1, y1, x2, y2 = bbox.tolist()
        poly_shapes = [{
            'label':      label,
            'points':     [[float(x1), float(y1)], [float(x2), float(y1)],
                           [float(x2), float(y2)], [float(x1), float(y2)]],
            'group_id':   None,
            'shape_type': 'polygon',
            'flags':      {},
        }]

    poly_json = {
        'version':     raw_data.get('version', '0.4.29'),
        'shapes':      poly_shapes,
        'imagePath':   f"{stem}.png",
        'imageHeight': H,
        'imageWidth':  W,
    }
    poly_path = out_poly_dir / f"{stem}.json"
    with open(poly_path, 'w', encoding='utf-8') as fh:
        json.dump(poly_json, fh, ensure_ascii=False, indent=2)

    # ---- Save review visualisation ---------------------------------------
    out_review_dir.mkdir(parents=True, exist_ok=True)
    right_title = 'SAM mask' if method == 'medsam' else 'BBox mask'
    review = make_review_image(img_rgb, bbox, mask,
                               title_left='BBox prompt',
                               title_right=right_title)
    review_path = out_review_dir / f"{stem}.jpg"
    cv2.imwrite(str(review_path), review, [cv2.IMWRITE_JPEG_QUALITY, 90])

    print(f"  [OK] {stem}  method={method}  "
          f"mask_px={mask_pos_px}  poly_pts={sum(len(s['points']) for s in poly_shapes)}")

    return {
        'stem':      stem,
        'method':    method,
        'mask_px':   mask_pos_px,
        'mask_path': str(mask_path),
        'poly_path': str(poly_path),
        'review':    str(review_path),
    }


# --------------------------------------------------------------------------- #
# 7.  Batch orchestration                                                     #
# --------------------------------------------------------------------------- #

def collect_pairs(data_root: Path) -> List[Dict[str, Path]]:
    """Collect all (image, annotation, mask_dir) triples from ``data_root``.

    Scans ``WLI/`` and ``NBI/`` subdirectories for PNG images and matches
    each to the corresponding annotation JSON in ``polygon_annotations/``.

    Args:
        data_root: Root of ``processed_data/``.

    Returns:
        List of dicts with keys ``'img'``, ``'json'``, ``'mask_dir'``.
    """
    pairs = []
    poly_dir = data_root / 'polygon_annotations'

    for modality in ('WLI', 'NBI'):
        img_dir = data_root / modality
        if not img_dir.exists():
            continue
        mask_dir = data_root / f"Mask_{modality}"
        for img_path in sorted(img_dir.glob('*.png')):
            json_path = poly_dir / f"{img_path.stem}.json"
            if not json_path.exists():
                print(f"[WARN] No annotation JSON for {img_path.name}, skipping")
                continue
            pairs.append({
                'img':      img_path,
                'json':     json_path,
                'mask_dir': mask_dir,
            })
    return pairs


def filter_pairs_by_cases(
    pairs: List[Dict[str, Path]], cases: List[str]
) -> List[Dict[str, Path]]:
    """Keep only pairs whose stem starts with one of the given case IDs.

    Args:
        pairs:  Full list of pairs.
        cases:  List of case IDs to keep (e.g. ``['case001', 'case002']``).

    Returns:
        Filtered list.
    """
    return [p for p in pairs
            if any(p['img'].stem.startswith(c) for c in cases)]


# --------------------------------------------------------------------------- #
# 8.  Entry point                                                             #
# --------------------------------------------------------------------------- #

def main() -> None:
    """CLI entry point for the MedSAM annotation pipeline."""
    parser = argparse.ArgumentParser(
        description='MedSAM bbox-to-mask refinement pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--data_root', default='processed_data',
                        help='Root of processed_data/ from preprocess.py')
    parser.add_argument('--sam_checkpoint', default=None,
                        help='Path to SAM/MedSAM .pth checkpoint')
    parser.add_argument('--model_type', default='vit_b',
                        choices=['vit_b', 'vit_l', 'vit_h'],
                        help='SAM model variant (MedSAM official uses vit_b)')
    parser.add_argument('--device', default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Inference device')
    parser.add_argument('--batch', action='store_true',
                        help='Process all images in data_root (WLI + NBI)')
    parser.add_argument('--cases', nargs='*', default=None,
                        help='Only process specific case IDs')
    parser.add_argument('--no_sam', action='store_true',
                        help='Skip SAM; generate bbox masks only (pipeline test)')
    parser.add_argument('--min_area', type=int, default=50,
                        help='Min connected-component area to keep (px)')
    parser.add_argument('--epsilon', type=float, default=0.005,
                        help='DP simplification factor for polygon approximation')
    args = parser.parse_args()

    data_root   = Path(args.data_root)
    review_dir  = data_root / 'reviews'
    poly_dir    = data_root / 'polygon_annotations'

    if not data_root.exists():
        print(f"ERROR: --data_root '{data_root}' does not exist.", file=sys.stderr)
        sys.exit(1)

    # ---- SAM initialisation -----------------------------------------------
    predictor = None
    if not args.no_sam:
        if not SAM_AVAILABLE:
            print(
                "[WARN] 'segment_anything' package not found.\n"
                "       Install with:\n"
                "         pip install git+https://github.com/facebookresearch/"
                "segment-anything.git\n"
                "       Then download the MedSAM checkpoint from:\n"
                "         https://drive.google.com/drive/folders/"
                "1ETWmi4AiniJeWOt6HAsYgTjYv_fkgzoN\n"
                "       Falling back to pure bbox masks."
            )
        elif args.sam_checkpoint is None:
            print(
                "[WARN] --sam_checkpoint not provided.\n"
                "       Falling back to pure bbox masks.\n"
                "       Pass --no_sam to silence this warning."
            )
        else:
            ckpt = Path(args.sam_checkpoint)
            if not ckpt.exists():
                print(f"[ERROR] Checkpoint not found: {ckpt}", file=sys.stderr)
                sys.exit(1)

            import torch
            print(f"Loading SAM model ({args.model_type}) from {ckpt} …")
            model = sam_model_registry[args.model_type](checkpoint=str(ckpt))
            model.to(device=args.device)
            predictor = SamPredictor(model)
            print("SAM model ready.\n")

    if predictor is None and not args.no_sam:
        # Already warned above; just mark as no_sam mode
        args.no_sam = True

    # ---- Collect image-annotation pairs ----------------------------------
    if args.batch or args.cases:
        pairs = collect_pairs(data_root)
        if args.cases:
            pairs = filter_pairs_by_cases(pairs, args.cases)
    else:
        # Default: process everything
        pairs = collect_pairs(data_root)

    if not pairs:
        print("No image-annotation pairs found.")
        sys.exit(0)

    mode = 'no_sam (bbox)' if args.no_sam else f'MedSAM/{args.model_type}'
    print(f"Mode : {mode}")
    print(f"Found: {len(pairs)} image(s)")
    print(f"Output → {data_root.resolve()}\n")

    # ---- Main loop --------------------------------------------------------
    results = []
    for pair in pairs:
        modality = 'WLI' if '_WLI_' in pair['img'].stem else 'NBI'
        print(f"Processing {pair['img'].stem} …")
        r = process_image(
            img_path       = pair['img'],
            json_path      = pair['json'],
            out_mask_dir   = pair['mask_dir'],
            out_poly_dir   = poly_dir,
            out_review_dir = review_dir,
            predictor      = predictor,
            no_sam         = args.no_sam,
        )
        if r:
            results.append(r)

    # ---- Summary ----------------------------------------------------------
    total = len(results)
    sam_n = sum(1 for r in results if r.get('method') == 'medsam')
    bbox_n = total - sam_n
    avg_px = sum(r.get('mask_px', 0) for r in results) / max(1, total)

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total processed : {total}")
    print(f"  MedSAM masks  : {sam_n}")
    print(f"  BBox masks    : {bbox_n}")
    print(f"  Avg mask px   : {avg_px:.0f}")
    print(f"  Reviews saved : {review_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
