# CLAUDE.md — DSCMFNet Project Instructions

## Project Identity

- **Project name:** DSCMFNet (Dual-Stream Cross-Modal Fusion Network)
- **Task:** Pixel-level segmentation of Early Gastric Cancer (EGC) from paired WLI + NBI endoscopic images
- **Language:** Python 3.10+
- **Framework:** PyTorch 2.0+
- **Style:** Follow FCBFormer (`ESandML/FCBFormer`) code conventions

## Critical Domain Knowledge

EGC is NOT like colorectal polyps. Do NOT apply polyp-specific assumptions:
- EGC lesions are flat (0-IIb) or depressed (0-IIc), not protruding
- Boundaries are diffuse and fuzzy, not sharp edges
- Lesion-to-mucosa contrast is extremely low
- Single-modality (WLI alone) is insufficient; NBI reveals microvascular (VS) and microsurface (MS) abnormalities
- Clinical diagnosis flow: WLI distant scan to locate → NBI close-up zoom to characterize

## Data Description

- **47 confirmed EGC cases** from Songjiang Hospital
- Each case has 4 images: WLI distant (白光远景), WLI close-up (白光近景), NBI distant (NBI远景), NBI close-up (NBI近景)
- Original annotations: LabelMe JSON with `shape_type: "rectangle"` (bounding boxes, NOT segmentation masks)
- Image format: 1276×1020 JPEG with black borders + metadata overlay on left ~35%
- Masks generated via MedSAM with bbox prompt → doctor correction → binary PNG

### Data Directory Structure

```
raw_data/
├── case001/
│   ├── 014.jpg                  # WLI distant
│   ├── 014白光远景.json
│   ├── 023.jpg                  # WLI close-up
│   ├── 023白光近景.json
│   ├── 026.jpg                  # NBI distant
│   ├── 026NBI远景.json
│   ├── 029.jpg                  # NBI close-up
│   └── 029NBI近景.json
├── case002/
│   └── ...

processed_data/
├── WLI/                         # Cropped FOV images
│   ├── case001_WLI_distant.png
│   └── case001_WLI_closeup.png
├── NBI/
│   ├── case001_NBI_distant.png
│   └── case001_NBI_closeup.png
├── Mask_WLI/                    # Per-modality binary masks
│   └── case001_WLI_distant.png
├── Mask_NBI/
│   └── case001_NBI_closeup.png
└── polygon_annotations/         # LabelMe polygon JSONs for review
```

### LabelMe JSON Format

```json
{
  "version": "0.4.29",
  "shapes": [{
    "label": "NBI近景",           // or "白光远景", "白光近景", "NBI远景"
    "points": [[x1, y1], [x2, y2]],
    "shape_type": "rectangle"     // will become "polygon" after MedSAM
  }],
  "imagePath": "029.jpg",
  "imageHeight": 1020,
  "imageWidth": 1276
}
```

Label naming convention for classifying modality and view:
- Contains "NBI" or "nbi" → NBI modality; contains "白光" → WLI modality
- Contains "远景" → distant view; contains "近景" → close-up view

## Architecture Specification

### Overview

```
WLI image ──► PVTv2-B2 ──► [C1_w, C2_w, C3_w, C4_w]
                                │      │      │      │
                             concat  CMFIM  CMFIM  CMFIM   (at 1/4, 1/8, 1/16, 1/32)
                                │      │      │      │
NBI image ──► ResNet34 ──► [C1_n, C2_n, C3_n, C4_n]
                                │      │      │      │
                              [F1,    F2,    F3,    F4]  ← fused features
                                │      │      │      │
                          Progressive Decoder (bottom-up with skips)
                                │
                         ┌──────┴──────┐
                    Seg Head      Boundary Head
                     (1ch)          (1ch)
                         │
                   Deep Supervision (×3 aux heads)
```

### Encoder: Stream A — WLI Context (PVTv2-B2)

- Input: WLI image (B, 3, 352, 352)
- Backbone: PVTv2-B2 (Pyramid Vision Transformer v2)
  - 4 stages with Spatial-Reduction Attention (SRA)
  - Overlapping patch embeddings (stride 4, 2, 2, 2)
  - Depths: [3, 4, 6, 3], heads: [1, 2, 5, 8], sr_ratios: [8, 4, 2, 1]
  - MLP ratios: [8, 8, 4, 4]
  - DWConv in FFN
- Output channels: [64, 128, 320, 512] at spatial scales [H/4, H/8, H/16, H/32]
- Pretrained: ImageNet (future: GastroNet-5M)

### Encoder: Stream B — NBI Detail (ResNet34)

- Input: NBI image (B, 3, 352, 352)
- Backbone: ResNet34 (`torchvision.models.resnet34`)
  - Standard conv1 → bn1 → relu → maxpool → layer1-4
- Output channels: [64, 128, 256, 512] at spatial scales [H/4, H/8, H/16, H/32]
- Pretrained: ImageNet

### Fusion Strategy (Scale-Dependent)

**At 1/4 scale (88×88):** Simple fusion only
- Channel Projection: project both to 64d
- Concatenate → 1×1 Conv → 64d output
- NO spatial alignment, NO cross-attention (too expensive, overfitting risk)

**At 1/8, 1/16, 1/32 scales:** Full CMFIM, each containing:

1. **Channel Projection:** 1×1 Conv + BN + GELU per stream → unified dim
2. **Spatial Alignment Module (SAM):**
   - Input: concat(source_feat, target_feat) along channel dim
   - 3×3 Conv + BN + GELU → 3×3 Conv → 2-channel offset field (dx, dy)
   - CRITICAL: init offset conv weights to zero, multiply offset by 0.1
   - Apply offset via `F.grid_sample(source, base_grid + offset, mode='bilinear', padding_mode='border', align_corners=True)`
   - Warps NBI features toward WLI coordinate space
3. **Bidirectional Cross-Modal Attention:**
   - Direction 1: Q=WLI, K/V=aligned_NBI → WLI gets texture
   - Direction 2: Q=aligned_NBI, K/V=WLI → NBI gets context
   - Channel reduction factor: 4 (e.g., 128d → 32d for Q/K/V projection)
   - Attention: standard dot-product, no multi-head splitting needed for small data
   - Residual: enhanced = BN(original + cross_attn_output)
4. **Feature Fusion:** Concat(enhanced_WLI, enhanced_NBI) → 1×1 Conv + 3×3 Conv + BN + GELU → F_i

### Progressive Decoder

```
F4 (B, 512, H/32, W/32) → 1×1 Conv → 256d
  ↑2× + concat(F3) → [3×3 Conv + BN + GELU] ×2 → 128d
  ↑2× + concat(F2) → [3×3 Conv + BN + GELU] ×2 → 64d
  ↑2× + concat(F1) → [3×3 Conv + BN + GELU] ×2 → 64d
  ↑4× (bilinear) → 3×3 Conv + BN + GELU → 32d
```

Use `F.interpolate(mode='bilinear', align_corners=True)` for upsampling.
Handle size mismatch: if `x.shape[2:] != skip.shape[2:]`, interpolate x to match skip before concat.

### Segmentation Head

`nn.Conv2d(32, 1, kernel_size=1)` → raw logits (apply sigmoid during inference/metrics only)

### Boundary Head

```
32d → 3×3 Conv + BN + GELU → 16d → 3×3 Conv + BN + GELU → 8d → 1×1 Conv → 1d
```

GT boundary extraction (on-the-fly, no pre-computation):
```python
dilated = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
eroded = -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)
boundary_gt = (dilated - eroded > 0).float()
```

### Deep Supervision

3 auxiliary 1×1 Conv heads producing 1-channel logits at decoder intermediate scales.
During loss computation, upsample each auxiliary prediction to full resolution before computing loss against GT.

## Loss Function

```python
L_total = L_structure(seg_pred, mask)
        + 0.5 * L_boundary(bdy_pred, mask)
        + 0.3 * (0.5 * L_structure(ds1, mask) + 0.3 * L_structure(ds2, mask) + 0.2 * L_structure(ds3, mask))
```

### Structure Loss (Weighted BCE + Weighted IoU)

```python
def structure_loss(pred, mask):
    # pred: (B, 1, H, W) logits; mask: (B, 1, H, W) binary
    # Upsample pred to mask size if different
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2,3)) / weit.sum(dim=(2,3))
    pred_sig = torch.sigmoid(pred)
    inter = ((pred_sig * mask) * weit).sum(dim=(2,3))
    union = ((pred_sig + mask) * weit).sum(dim=(2,3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()
```

### Boundary Loss

`F.binary_cross_entropy_with_logits(bdy_pred, boundary_gt)` where boundary_gt is computed from mask on-the-fly.

## Data Pipeline

### Preprocessing Script (`preprocess.py`)

1. **FOV Detection:** For Olympus GIF-H290Z images, detect non-black FOV region:
   - Convert to grayscale, threshold at 30
   - For each column in middle 70% of rows, compute bright pixel fraction
   - Smooth with kernel_size=15 moving average
   - FOV left edge = first column with smoothed brightness > 0.5
   - Same logic for top/bottom edges
2. **Crop** image to FOV bounding box
3. **Transform** bbox coordinates: subtract FOV origin
4. **Generate** binary mask from bbox (interim) or polygon (after MedSAM)
5. **Save** to processed_data/ directory structure

### MedSAM Annotator (`sam_annotator.py`)

- Use MedSAM (not vanilla SAM) for medical image segmentation
- Input: cropped image + transformed bbox as prompt
- Also add center point of bbox as positive point prompt
- Generate 3 candidate masks (`multimask_output=True`)
- Selection criteria: SAM confidence × 0.5 + bbox containment × 0.3 - area_ratio penalty - connectivity penalty
- Post-process: remove components < 50px area, fill holes, morphological close (5×5 ellipse, 2 iterations) then open (1 iteration)
- Export: binary mask PNG + LabelMe polygon JSON + review visualization (side-by-side image+bbox vs image+mask_overlay+contour)

### Dataloader — Phase 1: Same-Scale (`data/dataloader.py`)

- Pairs: WLI close-up + NBI close-up from same case
- Single shared mask (NBI close-up mask is primary GT)
- Albumentations synchronized geometric augmentation using `additional_targets={'nbi': 'image', 'mask': 'mask'}`
- Independent color augmentation per modality
- Resize to 352×352
- ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- Return dict: `{'wli', 'nbi', 'mask', 'case_id'}`

### Dataloader — Phase 2: Cross-Scale (`data/dataloader_crossscale.py`)

- Pairs: WLI distant + NBI close-up from same case
- Each modality has its OWN mask (different FOV = different GT)
- INDEPENDENT geometric augmentation per modality (cannot synchronize across different FOVs)
- WLI augment: lighter (flip, small rotate, mild color jitter)
- NBI augment: heavier (elastic deform, Gaussian noise, stronger rotate)
- Return dict: `{'wli', 'nbi', 'wli_mask', 'nbi_mask', 'case_id'}`

### File Naming Convention

Pattern: `{case_id}_{MODALITY}_{view}.png`
Examples: `case001_WLI_distant.png`, `case001_NBI_closeup.png`

Extract case_id with regex: `^(.+?)_{WLI|NBI}_{distant|closeup}$`
Fallback for simple naming: stem without underscores = case_id

## Training Script (`train.py`)

### Features Required

- 5-fold cross-validation loop (sklearn `KFold`)
- Mixed precision training (`torch.cuda.amp`)
- Cosine annealing LR with linear warm-up (10 epochs)
- Gradient clipping `max_norm=1.0`
- TensorBoard logging per fold
- Best model checkpointing (by validation Dice)
- Per-fold and cross-fold summary statistics (mean ± std)
- Support both Phase 1 and Phase 2 dataloaders via CLI flag

### CLI Arguments

```
--data_root       Path to processed_data/
--img_size        Input resolution (default: 352)
--epochs          Training epochs (default: 200)
--batch_size      Batch size (default: 4)
--lr              Learning rate (default: 1e-4)
--weight_decay    Weight decay (default: 1e-4)
--freeze_stages   Number of encoder stages to freeze (default: 2)
--lambda_bdy      Boundary loss weight (default: 0.5)
--alpha_ds        Deep supervision weight (default: 0.3)
--kfold           Number of folds (default: 5, set 0 for simple split)
--seed            Random seed (default: 42)
--num_workers     DataLoader workers (default: 0 for Windows compat)
--amp             Enable mixed precision (flag)
--output          Output directory (default: ./output)
--phase           1 or 2 (selects dataloader and augmentation strategy)
--print_freq      Print every N epochs (default: 10)
```

### Metrics Function

```python
def compute_metrics(pred_sigmoid, mask, threshold=0.5):
    # Returns dict: {'dice', 'iou', 'sensitivity', 'specificity', 'precision'}
    # Use eps=1e-8 for numerical stability
```

### Encoder Freezing

```python
def freeze_encoder_stages(model, num_stages):
    # PVTv2: freeze [patch_embed_i, block_i, norm_i] for i in range(num_stages)
    # ResNet34: stage 0 = [conv1, bn1, layer1], stage 1 = [layer2], stage 2 = [layer3], stage 3 = [layer4]
    # Set param.requires_grad = False
```

### Reproducibility

```python
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

## Ablation Experiment Plan

Code should support running these experiments by toggling CLI flags:

| # | Experiment | Config |
|---|-----------|--------|
| 1 | NBI-only baseline | Single ResNet34, NBI close-up input only |
| 2 | WLI-only baseline | Single PVTv2-B2, WLI close-up input only |
| 3 | Early fusion (6ch) | Concatenate WLI+NBI → single PVTv2-B2 (modify input conv to 6ch) |
| 4 | Dual-stream + concat | Phase 1: two encoders, concat fusion at all scales |
| 5 | + CMFIM | Phase 1 + full CMFIM at 1/8, 1/16, 1/32 |
| 6 | + SAM | + Spatial Alignment Module |
| 7 | + Boundary Head | + Boundary head and loss |
| 8 | Cross-scale | Phase 2: WLI distant + NBI close-up |
| 9 | + WLI aux loss | + Auxiliary WLI mask supervision |

## Implementation Priorities

1. **First:** `preprocess.py` — must work end-to-end on the actual data format
2. **Second:** `sam_annotator.py` — MedSAM integration with review export
3. **Third:** `data/dataloader.py` — Phase 1 same-scale loader
4. **Fourth:** `models/dscmfnet.py` — start with concat-only fusion, add CMFIM incrementally
5. **Fifth:** `train.py` — training loop with K-fold CV
6. **Sixth:** `data/dataloader_crossscale.py` — Phase 2 loader
7. **Last:** Ablation experiment configs

## Code Quality Requirements

- Type hints on all function signatures
- Docstrings (Google style) on all classes and public methods
- Chinese comments are acceptable for domain-specific clinical notes
- `num_workers=0` as default (Windows compatibility)
- All file I/O should handle both `.png` and `.jpg` extensions
- Print informative progress messages during training
- Never hardcode absolute paths; use CLI arguments or relative paths

## Key Libraries

```
torch >= 2.0
torchvision
albumentations
segment-anything  # or MedSAM fork
opencv-python
scikit-learn       # for KFold
tensorboard
numpy
Pillow
```

## Windows Compatibility Notes

- Default `num_workers=0` to avoid multiprocessing page file errors
- Use `if __name__ == '__main__':` guard in all entry points
- Prefer `os.path.join()` over f-strings with `/` for paths
