"""data/dataloader.py — Phase 1 same-scale paired dataloader for DSCMFNet.

Pairs WLI close-up + NBI close-up images from the same case and applies:
  * Synchronised geometric augmentation (shared random state for both images
    and the segmentation mask via Albumentations ``additional_targets``).
  * Independent per-modality colour augmentation (ColorJitter).
  * ImageNet normalisation → float32 tensors.

Public API::

    dataset = EGCPhase1Dataset(pairs, img_size=352, augment=True)
    train_loader, val_loader = build_kfold_loaders(
        data_root='processed_data', fold_idx=0, n_folds=5, batch_size=4)
"""

import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset

# --------------------------------------------------------------------------- #
# Constants                                                                    #
# --------------------------------------------------------------------------- #

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

_CASE_RE = re.compile(r'^(.+?)_(WLI|NBI)_(closeup|distant)$')


# --------------------------------------------------------------------------- #
# 1.  Pair discovery                                                           #
# --------------------------------------------------------------------------- #

def find_pairs(data_root: Path) -> List[Dict[str, object]]:
    """Discover valid WLI-close-up / NBI-close-up / Mask-NBI triplets.

    Scans ``data_root/WLI/`` for ``*_WLI_closeup.png`` files, then checks
    that the corresponding NBI image and NBI mask exist.

    Args:
        data_root: Root of ``processed_data/``.

    Returns:
        List of dicts, each with keys:

        ``case_id`` (str), ``wli`` (`Path`), ``nbi`` (`Path`),
        ``mask`` (`Path`).

        Prints a warning for every case where any file is missing.
    """
    wli_dir  = data_root / 'WLI'
    nbi_dir  = data_root / 'NBI'
    mask_dir = data_root / 'Mask_NBI'

    if not wli_dir.exists():
        raise FileNotFoundError(f"WLI directory not found: {wli_dir}")

    pairs: List[Dict[str, object]] = []
    missing = 0

    for wli_path in sorted(wli_dir.glob('*_WLI_closeup.png')):
        m = _CASE_RE.match(wli_path.stem)
        if m is None:
            warnings.warn(f"Unexpected filename pattern, skipping: {wli_path.name}")
            continue
        case_id = m.group(1)

        nbi_path  = nbi_dir  / f"{case_id}_NBI_closeup.png"
        mask_path = mask_dir / f"{case_id}_NBI_closeup.png"

        absent = [p for p in (nbi_path, mask_path) if not p.exists()]
        if absent:
            warnings.warn(
                f"[{case_id}] missing files: "
                + ", ".join(p.name for p in absent)
                + " — skipping this case."
            )
            missing += 1
            continue

        pairs.append({
            'case_id': case_id,
            'wli':  wli_path,
            'nbi':  nbi_path,
            'mask': mask_path,
        })

    print(f"[dataloader] Found {len(pairs)} valid case pair(s)"
          + (f"  ({missing} skipped due to missing files)" if missing else ""))
    return pairs


# --------------------------------------------------------------------------- #
# 2.  Augmentation pipelines                                                  #
# --------------------------------------------------------------------------- #

def _build_geo_aug(img_size: int) -> A.Compose:
    """Synchronised geometric augmentation applied to WLI, NBI, and mask.

    Args:
        img_size: Target spatial resolution (square).

    Returns:
        ``A.Compose`` with ``additional_targets={'nbi': 'image'}`` so that
        both modality images and the shared mask receive the same random
        spatial transforms.
    """
    return A.Compose(
        [
            A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR,
                     mask_interpolation=cv2.INTER_NEAREST),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Affine(
                translate_percent={'x': (-0.05, 0.05), 'y': (-0.05, 0.05)},
                scale=(0.90, 1.10),
                rotate=(-30, 30),
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.7,
            ),
            A.ElasticTransform(
                alpha=30, sigma=5,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.3,
            ),
        ],
        additional_targets={'nbi': 'image'},
    )


def _build_color_aug_wli() -> A.Compose:
    """Mild colour jitter for WLI (pinkish mucosa tones).

    WLI images have stable colour calibration; modest augmentation avoids
    creating unrealistic artefacts.
    """
    return A.Compose([
        A.ColorJitter(
            brightness=0.25, contrast=0.25,
            saturation=0.15, hue=0.05,
            p=0.8,
        ),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    ])


def _build_color_aug_nbi() -> A.Compose:
    """Stronger colour jitter for NBI (green-channel dominated images).

    NBI imaging amplifies microvascular patterns; broader augmentation
    improves robustness to scope-to-scope colour variation.
    """
    return A.Compose([
        A.ColorJitter(
            brightness=0.35, contrast=0.35,
            saturation=0.25, hue=0.08,
            p=0.8,
        ),
        A.GaussNoise(std_range=(0.01, 0.05), p=0.3),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
    ])


def _build_val_resize(img_size: int) -> A.Compose:
    """Validation transform: resize only, no geometric randomness."""
    return A.Compose(
        [A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR,
                  mask_interpolation=cv2.INTER_NEAREST)],
        additional_targets={'nbi': 'image'},
    )


def _normalize_to_tensor(img: np.ndarray) -> torch.Tensor:
    """Apply ImageNet normalisation and convert HWC → CHW float32 tensor.

    Args:
        img: uint8 HxWx3 RGB image.

    Returns:
        Float32 CHW tensor in roughly ``[-2.1, 2.6]`` (ImageNet stats).
    """
    pipe = A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    return pipe(image=img)['image'].float()


# --------------------------------------------------------------------------- #
# 3.  Dataset                                                                 #
# --------------------------------------------------------------------------- #

class EGCPhase1Dataset(Dataset):
    """Phase 1 same-scale paired dataset for DSCMFNet.

    Each sample is a WLI close-up + NBI close-up pair from the same
    endoscopy case, sharing a single binary segmentation mask derived from
    the NBI annotation.

    Args:
        pairs:    List of pair dicts as returned by :func:`find_pairs`.
        img_size: Spatial resolution after resizing (default 352).
        augment:  If ``True``, apply random geometric + colour augmentation.
                  Set ``False`` for validation / inference.
    """

    def __init__(
        self,
        pairs: List[Dict[str, object]],
        img_size: int = 352,
        augment: bool = True,
    ) -> None:
        self.pairs    = pairs
        self.img_size = img_size
        self.augment  = augment

        self._geo_aug      = _build_geo_aug(img_size) if augment else _build_val_resize(img_size)
        self._color_wli    = _build_color_aug_wli()   if augment else None
        self._color_nbi    = _build_color_aug_nbi()   if augment else None

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.pairs)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_rgb(path: Path) -> np.ndarray:
        """Load an image as uint8 HxWx3 RGB.

        Args:
            path: Path to the image file (.png or .jpg).

        Returns:
            uint8 RGB array.

        Raises:
            FileNotFoundError: If the file does not exist.
            OSError: If cv2 cannot decode the file.
        """
        img = cv2.imread(str(path))
        if img is None:
            raise OSError(f"cv2.imread failed: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _load_mask(path: Path) -> np.ndarray:
        """Load a binary mask as uint8 HxW with values 0 / 1.

        Args:
            path: Path to the mask PNG.

        Returns:
            uint8 array with 0 (background) and 1 (lesion).
        """
        m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise OSError(f"cv2.imread failed for mask: {path}")
        return (m > 127).astype(np.uint8)

    # ------------------------------------------------------------------ #
    def __getitem__(self, idx: int) -> Dict[str, object]:
        """Return one training sample.

        Args:
            idx: Index into ``self.pairs``.

        Returns:
            Dict with keys:

            * ``'wli'``     — float32 CHW tensor (3, H, W)
            * ``'nbi'``     — float32 CHW tensor (3, H, W)
            * ``'mask'``    — float32 tensor (1, H, W) with values 0 / 1
            * ``'case_id'`` — str
        """
        pair = self.pairs[idx]
        case_id  = pair['case_id']
        wli_path = pair['wli']
        nbi_path = pair['nbi']
        msk_path = pair['mask']

        wli  = self._load_rgb(wli_path)
        nbi  = self._load_rgb(nbi_path)
        mask = self._load_mask(msk_path)

        # ---- Synchronised geometric augmentation (WLI + NBI + mask) ------
        geo_out = self._geo_aug(image=wli, nbi=nbi, mask=mask)
        wli  = geo_out['image']
        nbi  = geo_out['nbi']
        mask = geo_out['mask']

        # ---- Independent colour augmentation ------------------------------
        if self._color_wli is not None:
            wli = self._color_wli(image=wli)['image']
        if self._color_nbi is not None:
            nbi = self._color_nbi(image=nbi)['image']

        # ---- Normalise + convert to tensor --------------------------------
        wli_t  = _normalize_to_tensor(wli)                              # (3,H,W)
        nbi_t  = _normalize_to_tensor(nbi)                              # (3,H,W)
        mask_t = torch.from_numpy(mask).unsqueeze(0).float()            # (1,H,W)

        return {
            'wli':     wli_t,
            'nbi':     nbi_t,
            'mask':    mask_t,
            'case_id': case_id,
        }


# --------------------------------------------------------------------------- #
# 4.  K-fold loader factory                                                   #
# --------------------------------------------------------------------------- #

def build_kfold_loaders(
    data_root: str | Path,
    fold_idx: int,
    n_folds:    int = 5,
    batch_size: int = 4,
    img_size:   int = 352,
    seed:       int = 42,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """Build train / validation DataLoaders for one fold of K-fold CV.

    Args:
        data_root:   Root of ``processed_data/``.
        fold_idx:    Zero-based fold index (must be < ``n_folds``).
        n_folds:     Total number of folds.
        batch_size:  Samples per mini-batch.
        img_size:    Spatial resolution fed to the network (default 352).
        seed:        Random seed for reproducible splits.
        num_workers: DataLoader worker processes (default 0 for Windows compat).

    Returns:
        ``(train_loader, val_loader)`` for the requested fold.

    Raises:
        ValueError: If ``fold_idx >= n_folds`` or no valid pairs are found.
    """
    if fold_idx >= n_folds:
        raise ValueError(f"fold_idx={fold_idx} must be < n_folds={n_folds}")

    data_root = Path(data_root)
    all_pairs = find_pairs(data_root)

    if not all_pairs:
        raise ValueError(f"No valid pairs found in {data_root}")

    if len(all_pairs) < n_folds:
        warnings.warn(
            f"Only {len(all_pairs)} case(s) available for {n_folds}-fold CV. "
            "Consider reducing --kfold."
        )

    indices = np.arange(len(all_pairs))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    splits = list(kf.split(indices))
    train_idx, val_idx = splits[fold_idx]

    train_pairs = [all_pairs[i] for i in train_idx]
    val_pairs   = [all_pairs[i] for i in val_idx]

    print(f"[fold {fold_idx}/{n_folds}] "
          f"train={len(train_pairs)}  val={len(val_pairs)}")

    train_ds = EGCPhase1Dataset(train_pairs, img_size=img_size, augment=True)
    val_ds   = EGCPhase1Dataset(val_pairs,   img_size=img_size, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=len(train_ds) >= batch_size,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


# --------------------------------------------------------------------------- #
# 5.  Smoke-test                                                              #
# --------------------------------------------------------------------------- #

def _make_dummy_data(root: Path, n_cases: int = 4) -> None:
    """Create a minimal dummy processed_data structure for testing.

    Generates random 256×256 PNG images and binary masks for
    ``n_cases`` cases so that the dataloader can be exercised without
    real clinical data.

    Args:
        root:    Directory to create (will be made if absent).
        n_cases: Number of fake case pairs to generate.
    """
    import random

    rng = np.random.default_rng(0)
    for i in range(1, n_cases + 1):
        cid = f"case{i:03d}"
        H, W = 256, 256

        for modality, subdir in [('WLI', 'WLI'), ('NBI', 'NBI')]:
            img_dir = root / subdir
            img_dir.mkdir(parents=True, exist_ok=True)
            img = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
            cv2.imwrite(str(img_dir / f"{cid}_{modality}_closeup.png"), img)

        for mask_subdir in ('Mask_WLI', 'Mask_NBI'):
            msk_dir = root / mask_subdir
            msk_dir.mkdir(parents=True, exist_ok=True)
            # Random binary blob mask
            mask = np.zeros((H, W), dtype=np.uint8)
            cy, cx = rng.integers(64, 192, size=2)
            ry, rx = rng.integers(20, 60, size=2)
            yy, xx = np.ogrid[:H, :W]
            mask[((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1] = 255
            cv2.imwrite(
                str(msk_dir / f"{cid}_{mask_subdir.split('_')[1]}_closeup.png"),
                mask,
            )


def _smoke_test(data_root: str = "dummy_data") -> None:
    """Run a quick end-to-end smoke test of the dataloader.

    Creates a temporary dummy dataset, iterates one batch, and prints
    tensor shapes and value ranges.

    Args:
        data_root: Directory for dummy data (created and left in place).
    """
    root = Path(data_root)
    print(f"\n{'='*60}")
    print("Smoke test — generating dummy data …")
    _make_dummy_data(root, n_cases=4)

    pairs = find_pairs(root)
    if not pairs:
        print("[ERROR] No pairs found in dummy data.")
        return

    # ---- Training dataset (augment=True) ----------------------------------
    ds = EGCPhase1Dataset(pairs, img_size=352, augment=True)
    loader = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0)

    batch = next(iter(loader))
    print("\n--- Batch shapes and value ranges ---")
    for key in ('wli', 'nbi', 'mask'):
        t = batch[key]
        print(f"  {key:6s}: shape={tuple(t.shape)}  "
              f"dtype={t.dtype}  "
              f"min={t.min():.3f}  max={t.max():.3f}")
    print(f"  case_ids : {batch['case_id']}")

    # ---- K-fold loaders (2-fold given only 4 cases) -----------------------
    print("\n--- build_kfold_loaders (n_folds=2, fold_idx=0) ---")
    train_loader, val_loader = build_kfold_loaders(
        data_root=root, fold_idx=0, n_folds=2, batch_size=2)
    print(f"  train batches: {len(train_loader)}  "
          f"val batches: {len(val_loader)}")

    # Iterate val loader completely
    for vbatch in val_loader:
        assert vbatch['wli'].shape[1:] == (3, 352, 352)
        assert vbatch['mask'].shape[1:] == (1, 352, 352)

    print("\nSmoke test PASSED.")
    print("=" * 60)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Dataloader smoke test')
    parser.add_argument('--data_root', default='dummy_data',
                        help='Dummy data directory')
    args = parser.parse_args()
    _smoke_test(args.data_root)
