"""data/dataloader_crossscale.py — Phase 2 cross-scale paired dataloader.

Pairs **WLI distant** + **NBI close-up** from the same case.  Because the
two modalities capture different fields of view (FOV), geometric augmentation
is applied **independently** per modality — synchronising transforms across
different FOVs would introduce label noise.

Key differences from Phase 1 (:mod:`data.dataloader`):

* Pairing: ``WLI/case_WLI_distant.png`` + ``NBI/case_NBI_closeup.png``
* Separate GT masks: ``Mask_WLI/case_WLI_distant.png`` (WLI FOV) and
  ``Mask_NBI/case_NBI_closeup.png`` (NBI FOV)
* Independent geometric augmentation per modality:
  - WLI distant — lighter (small flip, mild rotate/scale)
  - NBI close-up — heavier (elastic, strong rotate, Gaussian noise)
* Primary GT for training loss: ``nbi_mask`` (NBI annotation is the
  diagnostic reference for EGC characterisation)

Public API::

    dataset = EGCPhase2Dataset(pairs, img_size=352, augment=True)
    train_loader, val_loader = build_kfold_loaders(
        data_root='processed_data', fold_idx=0, n_folds=5, batch_size=4)
"""

import re
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset

from data.dataloader import IMAGENET_MEAN, IMAGENET_STD

# --------------------------------------------------------------------------- #
# Constants                                                                   #
# --------------------------------------------------------------------------- #

_CASE_RE = re.compile(r'^(.+?)_(WLI|NBI)_(closeup|distant)$')


# --------------------------------------------------------------------------- #
# 1.  Pair discovery                                                          #
# --------------------------------------------------------------------------- #

def find_crossscale_pairs(data_root: Path) -> List[Dict[str, object]]:
    """Discover valid WLI-distant / NBI-closeup / dual-mask quadruplets.

    Scans ``data_root/WLI/`` for ``*_WLI_distant.png`` files, then
    checks that the corresponding NBI close-up image and both masks exist.

    Args:
        data_root: Root of ``processed_data/``.

    Returns:
        List of dicts, each with keys:

        ``case_id`` (str), ``wli`` (`Path`), ``nbi`` (`Path`),
        ``wli_mask`` (`Path`), ``nbi_mask`` (`Path`).
    """
    wli_dir      = data_root / 'WLI'
    nbi_dir      = data_root / 'NBI'
    wli_mask_dir = data_root / 'Mask_WLI'
    nbi_mask_dir = data_root / 'Mask_NBI'

    if not wli_dir.exists():
        raise FileNotFoundError(f"WLI directory not found: {wli_dir}")

    pairs: List[Dict[str, object]] = []
    missing = 0

    for wli_path in sorted(wli_dir.glob('*_WLI_distant.png')):
        m = _CASE_RE.match(wli_path.stem)
        if m is None:
            warnings.warn(f"Unexpected filename, skipping: {wli_path.name}")
            continue
        case_id = m.group(1)

        nbi_path      = nbi_dir      / f"{case_id}_NBI_closeup.png"
        wli_mask_path = wli_mask_dir / f"{case_id}_WLI_distant.png"
        nbi_mask_path = nbi_mask_dir / f"{case_id}_NBI_closeup.png"

        absent = [p for p in (nbi_path, wli_mask_path, nbi_mask_path)
                  if not p.exists()]
        if absent:
            warnings.warn(
                f"[{case_id}] missing cross-scale files: "
                + ", ".join(p.name for p in absent)
                + " — skipping this case."
            )
            missing += 1
            continue

        pairs.append({
            'case_id':  case_id,
            'wli':      wli_path,
            'nbi':      nbi_path,
            'wli_mask': wli_mask_path,
            'nbi_mask': nbi_mask_path,
        })

    print(f"[crossscale loader] Found {len(pairs)} valid cross-scale pair(s)"
          + (f"  ({missing} skipped)" if missing else ""))
    return pairs


# --------------------------------------------------------------------------- #
# 2.  Augmentation pipelines                                                  #
# --------------------------------------------------------------------------- #
# Phase 2 uses INDEPENDENT geo aug per modality — no shared random state.
# Each pipeline receives (image=img, mask=mask_arr) and returns both.

def _build_wli_geo_aug(img_size: int) -> A.Compose:
    """Lighter geometric aug for WLI distant view.

    Distant scans cover a wide FOV; heavy spatial distortion would alter
    the stomach topology.  Small rotate/scale/translate is sufficient.
    """
    return A.Compose([
        A.Resize(img_size, img_size,
                 interpolation=cv2.INTER_LINEAR,
                 mask_interpolation=cv2.INTER_NEAREST),
        A.HorizontalFlip(p=0.5),
        A.Affine(
            translate_percent={'x': (-0.05, 0.05), 'y': (-0.05, 0.05)},
            scale=(0.95, 1.05),
            rotate=(-15, 15),
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.6,
        ),
    ])


def _build_nbi_geo_aug(img_size: int) -> A.Compose:
    """Heavier geometric aug for NBI close-up view.

    Close-up scans are taken at varying angles and distances; elastic
    deformation and larger rotation simulate realistic endoscope movement.
    """
    return A.Compose([
        A.Resize(img_size, img_size,
                 interpolation=cv2.INTER_LINEAR,
                 mask_interpolation=cv2.INTER_NEAREST),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Affine(
            translate_percent={'x': (-0.08, 0.08), 'y': (-0.08, 0.08)},
            scale=(0.85, 1.15),
            rotate=(-30, 30),
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.7,
        ),
        A.ElasticTransform(
            alpha=50, sigma=8,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.4,
        ),
    ])


def _build_wli_color_aug() -> A.Compose:
    """Mild colour jitter for WLI distant (stable white-light illumination)."""
    return A.Compose([
        A.ColorJitter(
            brightness=0.2, contrast=0.2,
            saturation=0.1, hue=0.03,
            p=0.7,
        ),
        A.GaussianBlur(blur_limit=(3, 5), p=0.15),
    ])


def _build_nbi_color_aug() -> A.Compose:
    """Strong colour jitter + noise for NBI close-up (scope-to-scope variation)."""
    return A.Compose([
        A.ColorJitter(
            brightness=0.4, contrast=0.4,
            saturation=0.3, hue=0.08,
            p=0.8,
        ),
        A.GaussNoise(std_range=(0.01, 0.06), p=0.35),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
    ])


def _build_val_resize(img_size: int) -> A.Compose:
    """Validation / inference transform: resize only."""
    return A.Compose([
        A.Resize(img_size, img_size,
                 interpolation=cv2.INTER_LINEAR,
                 mask_interpolation=cv2.INTER_NEAREST),
    ])


def _normalize_to_tensor(img: np.ndarray) -> torch.Tensor:
    """ImageNet-normalise a uint8 HWC image and return a CHW float32 tensor."""
    pipe = A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    return pipe(image=img)['image'].float()


# --------------------------------------------------------------------------- #
# 3.  Dataset                                                                 #
# --------------------------------------------------------------------------- #

class EGCPhase2Dataset(Dataset):
    """Phase 2 cross-scale paired dataset (WLI distant + NBI close-up).

    Each sample provides:

    * WLI distant image — wide FOV, used by the PVTv2-B2 context stream.
    * NBI close-up image — narrow FOV, used by the ResNet34 detail stream.
    * Two independent binary masks (one per modality / FOV).

    Geometric augmentation is applied **independently** to each
    ``(image, mask)`` pair because the FOVs are different.  Colour
    augmentation is applied only to images (masks are unaffected).

    Args:
        pairs:    List of cross-scale pair dicts from
                  :func:`find_crossscale_pairs`.
        img_size: Spatial resolution after resize (default 352).
        augment:  Apply random augmentation when ``True`` (train);
                  resize-only when ``False`` (val / inference).
    """

    def __init__(
        self,
        pairs:    List[Dict[str, object]],
        img_size: int  = 352,
        augment:  bool = True,
    ) -> None:
        self.pairs    = pairs
        self.img_size = img_size
        self.augment  = augment

        if augment:
            self._wli_geo   = _build_wli_geo_aug(img_size)
            self._nbi_geo   = _build_nbi_geo_aug(img_size)
            self._wli_color = _build_wli_color_aug()
            self._nbi_color = _build_nbi_color_aug()
        else:
            self._val_resize = _build_val_resize(img_size)

    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self.pairs)

    # ------------------------------------------------------------------ #

    @staticmethod
    def _load_rgb(path: Path) -> np.ndarray:
        """Load an image as uint8 HxWx3 RGB."""
        img = cv2.imread(str(path))
        if img is None:
            raise OSError(f"cv2.imread failed: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _load_mask(path: Path) -> np.ndarray:
        """Load a binary mask as uint8 HxW with values 0 / 1."""
        m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise OSError(f"cv2.imread failed for mask: {path}")
        return (m > 127).astype(np.uint8)

    # ------------------------------------------------------------------ #

    def __getitem__(self, idx: int) -> Dict[str, object]:
        """Return one training/validation sample.

        Args:
            idx: Index into ``self.pairs``.

        Returns:
            Dict with keys:

            * ``'wli'``      — float32 ``(3, H, W)`` WLI distant tensor
            * ``'nbi'``      — float32 ``(3, H, W)`` NBI close-up tensor
            * ``'wli_mask'`` — float32 ``(1, H, W)`` WLI distant mask
            * ``'nbi_mask'`` — float32 ``(1, H, W)`` NBI close-up mask
            * ``'case_id'``  — str
        """
        pair    = self.pairs[idx]
        case_id = pair['case_id']

        wli      = self._load_rgb(pair['wli'])
        nbi      = self._load_rgb(pair['nbi'])
        wli_mask = self._load_mask(pair['wli_mask'])
        nbi_mask = self._load_mask(pair['nbi_mask'])

        if self.augment:
            # ---- WLI: geo (image+mask together) then colour (image only) --
            out      = self._wli_geo(image=wli, mask=wli_mask)
            wli      = out['image']
            wli_mask = out['mask']
            wli      = self._wli_color(image=wli)['image']

            # ---- NBI: independent geo (image+mask) then colour -------------
            out      = self._nbi_geo(image=nbi, mask=nbi_mask)
            nbi      = out['image']
            nbi_mask = out['mask']
            nbi      = self._nbi_color(image=nbi)['image']
        else:
            out      = self._val_resize(image=wli, mask=wli_mask)
            wli      = out['image']
            wli_mask = out['mask']

            out      = self._val_resize(image=nbi, mask=nbi_mask)
            nbi      = out['image']
            nbi_mask = out['mask']

        return {
            'wli':      _normalize_to_tensor(wli),
            'nbi':      _normalize_to_tensor(nbi),
            'wli_mask': torch.from_numpy(wli_mask).unsqueeze(0).float(),
            'nbi_mask': torch.from_numpy(nbi_mask).unsqueeze(0).float(),
            'case_id':  case_id,
        }


# --------------------------------------------------------------------------- #
# 4.  K-fold loader factory                                                   #
# --------------------------------------------------------------------------- #

def build_kfold_loaders(
    data_root:   'str | Path',
    fold_idx:    int,
    n_folds:     int = 5,
    batch_size:  int = 4,
    img_size:    int = 352,
    seed:        int = 42,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """Build train / validation DataLoaders for one fold of K-fold CV.

    Interface-compatible with the Phase 1
    :func:`data.dataloader.build_kfold_loaders`.

    Args:
        data_root:   Root of ``processed_data/``.
        fold_idx:    Zero-based fold index (must be ``< n_folds``).
        n_folds:     Total number of folds.
        batch_size:  Samples per mini-batch.
        img_size:    Spatial resolution fed to the network (default 352).
        seed:        Random seed for reproducible splits.
        num_workers: DataLoader worker count (default 0 for Windows compat).

    Returns:
        ``(train_loader, val_loader)`` for the requested fold.

    Raises:
        ValueError: If ``fold_idx >= n_folds`` or no valid pairs found.
    """
    if fold_idx >= n_folds:
        raise ValueError(f"fold_idx={fold_idx} must be < n_folds={n_folds}")

    data_root = Path(data_root)
    all_pairs = find_crossscale_pairs(data_root)

    if not all_pairs:
        raise ValueError(f"No valid cross-scale pairs found in {data_root}")

    if len(all_pairs) < n_folds:
        warnings.warn(
            f"Only {len(all_pairs)} case(s) for {n_folds}-fold CV. "
            "Consider reducing --kfold."
        )

    indices = np.arange(len(all_pairs))
    kf      = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    splits  = list(kf.split(indices))
    train_idx, val_idx = splits[fold_idx]

    train_pairs = [all_pairs[i] for i in train_idx]
    val_pairs   = [all_pairs[i] for i in val_idx]

    print(f"[crossscale fold {fold_idx}/{n_folds}] "
          f"train={len(train_pairs)}  val={len(val_pairs)}")

    train_ds = EGCPhase2Dataset(train_pairs, img_size=img_size, augment=True)
    val_ds   = EGCPhase2Dataset(val_pairs,   img_size=img_size, augment=False)

    kw = dict(num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        drop_last=len(train_ds) >= batch_size, **kw,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, **kw,
    )
    return train_loader, val_loader


# --------------------------------------------------------------------------- #
# 5.  Smoke test                                                               #
# --------------------------------------------------------------------------- #

def _smoke_test(data_root: str = 'processed_data') -> None:
    """Quick end-to-end test of the Phase 2 dataloader."""
    root = Path(data_root)

    print(f"\n{'='*60}")
    print("Phase 2 cross-scale dataloader smoke test")
    print(f"  data_root: {root.resolve()}")
    print(f"{'='*60}\n")

    # ---- Pair discovery -------------------------------------------------
    pairs = find_crossscale_pairs(root)
    if not pairs:
        print("[ERROR] No cross-scale pairs found.")
        return

    # ---- Single-sample augmentation check --------------------------------
    ds_aug = EGCPhase2Dataset(pairs, img_size=352, augment=True)
    ds_val = EGCPhase2Dataset(pairs, img_size=352, augment=False)

    sample_aug = ds_aug[0]
    sample_val = ds_val[0]

    print("--- Single sample shapes (augment=True) ---")
    for key in ('wli', 'nbi', 'wli_mask', 'nbi_mask'):
        t = sample_aug[key]
        print(f"  {key:9s}: shape={tuple(t.shape)}  "
              f"dtype={t.dtype}  "
              f"min={t.min():.3f}  max={t.max():.3f}")
    print(f"  case_id  : {sample_aug['case_id']}")

    # Shapes should be identical between aug and val after resize
    for key in ('wli', 'nbi', 'wli_mask', 'nbi_mask'):
        assert sample_aug[key].shape == sample_val[key].shape, \
            f"Shape mismatch on key '{key}'"

    # Masks are binary float [0, 1]
    assert sample_aug['wli_mask'].min() >= 0
    assert sample_aug['nbi_mask'].max() <= 1

    # ---- K-fold DataLoader -----------------------------------------------
    n_folds = min(2, len(pairs))
    print(f"\n--- build_kfold_loaders (n_folds={n_folds}, fold_idx=0) ---")
    train_loader, val_loader = build_kfold_loaders(
        data_root=root, fold_idx=0, n_folds=n_folds,
        batch_size=1, img_size=352, num_workers=0,
    )
    print(f"  train batches: {len(train_loader)}  val batches: {len(val_loader)}")

    for batch in val_loader:
        assert batch['wli'].shape[1:]      == (3, 352, 352)
        assert batch['nbi'].shape[1:]      == (3, 352, 352)
        assert batch['wli_mask'].shape[1:] == (1, 352, 352)
        assert batch['nbi_mask'].shape[1:] == (1, 352, 352)
        assert batch['wli_mask'].dtype     == torch.float32
        assert batch['nbi_mask'].dtype     == torch.float32

    print("\nSmoke test PASSED.")
    print("=" * 60)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Phase 2 dataloader smoke test')
    parser.add_argument('--data_root', default='processed_data')
    args = parser.parse_args()
    _smoke_test(args.data_root)
