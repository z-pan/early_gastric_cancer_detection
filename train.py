#!/usr/bin/env python3
"""train.py — DSCMFNet K-fold cross-validation training script.

Usage::

    # Full 5-fold training with DSCMFNet Phase 1 on real data
    python train.py --data_root processed_data --epochs 200 --kfold 5

    # Quick smoke test (2 cases, 1 epoch, 2 folds)
    python train.py --data_root processed_data --epochs 1 --kfold 2 \\
        --batch_size 1 --num_workers 0

    # Ablation: NBI-only baseline
    python train.py --data_root processed_data --model nbi_only --epochs 200

    # AMP + checkpoint
    python train.py --data_root processed_data --amp --output ./output/run1
"""

import argparse
import math
import random
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LambdaLR,
    LinearLR,
    SequentialLR,
)
from torch.utils.data import DataLoader

try:
    from torch.utils.tensorboard import SummaryWriter
    _TENSORBOARD = True
except ImportError:
    _TENSORBOARD = False

from data.dataloader import EGCPhase1Dataset, build_kfold_loaders, find_pairs
from models.dscmfnet import DSCMFNet, count_parameters
from models.losses import DSCMFNetLoss
from models.single_stream import (
    EarlyFusionPVTv2,
    SingleStreamPVTv2,
    SingleStreamResNet34,
)

# --------------------------------------------------------------------------- #
# Constants                                                                   #
# --------------------------------------------------------------------------- #

_WARMUP_EPOCHS = 10


# --------------------------------------------------------------------------- #
# 1.  Reproducibility                                                         #
# --------------------------------------------------------------------------- #

def seed_everything(seed: int) -> None:
    """Set all relevant random seeds for reproducible training.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------- #
# 2.  Metrics                                                                 #
# --------------------------------------------------------------------------- #

def compute_metrics(
    pred_sigmoid: torch.Tensor,
    mask: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute pixel-level segmentation metrics for a batch.

    Args:
        pred_sigmoid: ``(B, 1, H, W)`` sigmoid probabilities in [0, 1].
        mask:         ``(B, 1, H, W)`` binary float ground-truth mask.
        threshold:    Binarisation threshold (default 0.5).

    Returns:
        Dict with keys ``'dice'``, ``'iou'``, ``'sensitivity'``,
        ``'specificity'``, ``'precision'``.  All values are scalars
        averaged over the batch.
    """
    eps = 1e-8
    pred_bin = (pred_sigmoid > threshold).float()

    tp = (pred_bin * mask        ).sum(dim=(1, 2, 3))   # (B,)
    fp = (pred_bin * (1 - mask)  ).sum(dim=(1, 2, 3))
    fn = ((1 - pred_bin) * mask  ).sum(dim=(1, 2, 3))
    tn = ((1 - pred_bin) * (1 - mask)).sum(dim=(1, 2, 3))

    dice        = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou         = (tp + eps)     / (tp + fp + fn + eps)
    sensitivity = (tp + eps)     / (tp + fn + eps)
    specificity = (tn + eps)     / (tn + fp + eps)
    precision   = (tp + eps)     / (tp + fp + eps)

    return {
        'dice':        dice.mean().item(),
        'iou':         iou.mean().item(),
        'sensitivity': sensitivity.mean().item(),
        'specificity': specificity.mean().item(),
        'precision':   precision.mean().item(),
    }


# --------------------------------------------------------------------------- #
# 3.  Model factory                                                           #
# --------------------------------------------------------------------------- #

def build_model(
    model_type: str,
    pretrained: bool = True,
    fusion_mode: str = 'concat',
) -> nn.Module:
    """Instantiate the requested model variant.

    Args:
        model_type:  One of ``'nbi_only'``, ``'wli_only'``,
                     ``'early_fusion'``, ``'dscmfnet'``.
        pretrained:  Load ImageNet-pretrained encoder weights.
        fusion_mode: DSCMFNet-only — one of ``'concat'``, ``'cmfim'``,
                     ``'cmfim_no_sam'``, ``'cmfim_no_boundary'``.
                     Ignored for non-DSCMFNet models.

    Returns:
        Initialised :class:`torch.nn.Module`.
    """
    _map = {
        'nbi_only':     lambda: SingleStreamResNet34(pretrained=pretrained),
        'wli_only':     lambda: SingleStreamPVTv2(pretrained=pretrained),
        'early_fusion': lambda: EarlyFusionPVTv2(pretrained=pretrained),
        'dscmfnet':     lambda: DSCMFNet(pretrained=pretrained,
                                         fusion_mode=fusion_mode),
    }
    if model_type not in _map:
        raise ValueError(f"Unknown model type: {model_type!r}. "
                         f"Choose from {list(_map)}")
    return _map[model_type]()


def model_forward(
    model: nn.Module,
    wli: torch.Tensor,
    nbi: torch.Tensor,
    model_type: str,
) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
    """Dispatch the forward call to the correct model signature.

    Args:
        model:      Model instance.
        wli:        ``(B, 3, H, W)`` WLI tensor.
        nbi:        ``(B, 3, H, W)`` NBI tensor.
        model_type: String key matching :func:`build_model`.

    Returns:
        ``(seg_pred, bdy_pred, ds_preds)``
    """
    if model_type == 'nbi_only':
        return model(nbi)
    if model_type == 'wli_only':
        return model(wli)
    return model(wli, nbi)   # early_fusion, dscmfnet


# --------------------------------------------------------------------------- #
# 4.  Encoder freezing                                                        #
# --------------------------------------------------------------------------- #

def _freeze_resnet_stages(encoder: nn.Module, num_stages: int) -> None:
    """Freeze leading stages of a :class:`_ResNet34Encoder`.

    Stage mapping: 0→[stem, layer1], 1→[layer2], 2→[layer3], 3→[layer4].
    """
    groups = [
        [encoder.stem, encoder.layer1],
        [encoder.layer2],
        [encoder.layer3],
        [encoder.layer4],
    ]
    for i in range(min(num_stages, 4)):
        for mod in groups[i]:
            for p in mod.parameters():
                p.requires_grad = False


def _freeze_pvt_stages(backbone: nn.Module, num_stages: int) -> None:
    """Freeze leading stages of a timm PVTv2-B2 FeatureListNet backbone.

    Stage mapping (timm 1.x flat layout):
    0→[patch_embed, stages_0], 1→[stages_1], 2→[stages_2], 3→[stages_3].
    """
    groups = [
        [backbone.patch_embed, getattr(backbone, 'stages_0')],
        [getattr(backbone, 'stages_1')],
        [getattr(backbone, 'stages_2')],
        [getattr(backbone, 'stages_3')],
    ]
    for i in range(min(num_stages, 4)):
        for mod in groups[i]:
            for p in mod.parameters():
                p.requires_grad = False


def freeze_encoder_stages(model: nn.Module, num_stages: int) -> None:
    """Freeze the first ``num_stages`` encoder stages of any supported model.

    Dispatches to the per-architecture freeze logic.  When ``num_stages=0``
    this function is a no-op.

    Args:
        model:      Model instance (any variant built by :func:`build_model`).
        num_stages: Number of stages to freeze (0–4).
    """
    if num_stages == 0:
        return

    if isinstance(model, DSCMFNet):
        model.freeze_encoder_stages(num_stages)
    elif isinstance(model, SingleStreamResNet34):
        _freeze_resnet_stages(model.encoder, num_stages)
    elif isinstance(model, SingleStreamPVTv2):
        _freeze_pvt_stages(model.encoder.backbone, num_stages)
    elif isinstance(model, EarlyFusionPVTv2):
        _freeze_pvt_stages(model.encoder.backbone, num_stages)
    else:
        warnings.warn(
            f"freeze_encoder_stages: unrecognised model {type(model).__name__}, "
            "skipping freeze."
        )


# --------------------------------------------------------------------------- #
# 5.  LR schedule                                                             #
# --------------------------------------------------------------------------- #

def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    warmup: int = _WARMUP_EPOCHS,
) -> torch.optim.lr_scheduler._LRScheduler:
    """Build a linear warm-up + cosine annealing scheduler.

    When ``epochs <= warmup``, only a linear warm-up schedule is returned
    (avoids creating a cosine schedule with T_max=0).

    Args:
        optimizer: Bound optimiser.
        epochs:    Total number of training epochs.
        warmup:    Number of warm-up epochs (default 10).

    Returns:
        LR scheduler instance.
    """
    if epochs <= warmup:
        return LambdaLR(optimizer, lr_lambda=lambda e: (e + 1) / max(1, epochs))

    warmup_sched  = LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                             total_iters=warmup)
    cosine_sched  = CosineAnnealingLR(optimizer, T_max=epochs - warmup,
                                      eta_min=1e-6)
    return SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched],
                        milestones=[warmup])


# --------------------------------------------------------------------------- #
# 6.  Inner training / validation loops                                       #
# --------------------------------------------------------------------------- #

class _AverageMeter:
    """Lightweight running-average tracker."""
    __slots__ = ('_sum', '_n')

    def __init__(self) -> None:
        self._sum = 0.0
        self._n   = 0

    def update(self, val: float, n: int = 1) -> None:
        self._sum += val * n
        self._n   += n

    @property
    def avg(self) -> float:
        return self._sum / max(1, self._n)


def train_one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: DSCMFNetLoss,
    optimizer: torch.optim.Optimizer,
    scaler:    GradScaler,
    device:    torch.device,
    model_type: str,
    amp:       bool,
) -> Dict[str, float]:
    """Run one full training epoch.

    Args:
        model:      Model in train mode.
        loader:     Training DataLoader.
        criterion:  Loss function.
        optimizer:  Optimiser.
        scaler:     AMP grad scaler (no-op when ``amp=False``).
        device:     Computation device.
        model_type: Model type key for :func:`model_forward`.
        amp:        Whether to use mixed precision.

    Returns:
        Dict with key ``'loss'`` (epoch-average training loss).
    """
    model.train()
    loss_meter = _AverageMeter()

    for batch in loader:
        wli  = batch['wli' ].to(device, non_blocking=True)
        nbi  = batch['nbi' ].to(device, non_blocking=True)
        mask = batch['mask'].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=amp):
            seg, bdy, ds = model_forward(model, wli, nbi, model_type)
            loss = criterion(seg, bdy, ds, mask)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        loss_meter.update(loss.item(), n=wli.size(0))

    return {'loss': loss_meter.avg}


@torch.no_grad()
def validate(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: DSCMFNetLoss,
    device:    torch.device,
    model_type: str,
    amp:       bool,
) -> Dict[str, float]:
    """Run one full validation epoch and return loss + metrics.

    Args:
        model:      Model in eval mode.
        loader:     Validation DataLoader.
        criterion:  Loss function.
        device:     Computation device.
        model_type: Model type key.
        amp:        Whether to use mixed precision.

    Returns:
        Dict with keys ``'loss'``, ``'dice'``, ``'iou'``, ``'sensitivity'``,
        ``'specificity'``, ``'precision'``.
    """
    model.eval()
    meters: Dict[str, _AverageMeter] = {
        k: _AverageMeter()
        for k in ('loss', 'dice', 'iou', 'sensitivity', 'specificity', 'precision')
    }

    for batch in loader:
        wli  = batch['wli' ].to(device, non_blocking=True)
        nbi  = batch['nbi' ].to(device, non_blocking=True)
        mask = batch['mask'].to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=amp):
            seg, bdy, ds = model_forward(model, wli, nbi, model_type)
            loss = criterion(seg, bdy, ds, mask)

        m = compute_metrics(torch.sigmoid(seg), mask)
        n = wli.size(0)
        meters['loss'].update(loss.item(), n)
        for k, v in m.items():
            meters[k].update(v, n)

    return {k: v.avg for k, v in meters.items()}


# --------------------------------------------------------------------------- #
# 7.  Per-fold training                                                       #
# --------------------------------------------------------------------------- #

def train_fold(
    fold_idx:     int,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    args:         argparse.Namespace,
) -> Dict[str, float]:
    """Train and validate for one cross-validation fold.

    Args:
        fold_idx:     Zero-based fold index.
        train_loader: Training DataLoader for this fold.
        val_loader:   Validation DataLoader for this fold.
        args:         Parsed CLI arguments.

    Returns:
        Dict with ``'fold'``, ``'best_val_dice'``, ``'best_epoch'`` and
        averaged final validation metrics.
    """
    device = torch.device(args.device)
    out_dir = Path(args.output) / f'fold_{fold_idx}'
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Model ----------------------------------------------------------- #
    model = build_model(
        args.model,
        pretrained=args.pretrained,
        fusion_mode=args.fusion_mode,
    ).to(device)
    freeze_encoder_stages(model, args.freeze_stages)

    param_info = count_parameters(model)
    print(f"\n  [fold {fold_idx}] {args.model}  "
          f"total={param_info['total']:,}  "
          f"trainable={param_info['trainable']:,}")

    # ---- Optimiser ------------------------------------------------------- #
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = _build_scheduler(optimizer, args.epochs)
    criterion = DSCMFNetLoss(lambda_bdy=args.lambda_bdy, alpha_ds=args.alpha_ds)

    # ---- AMP ------------------------------------------------------------- #
    use_amp = args.amp and device.type == 'cuda'
    if args.amp and not use_amp:
        warnings.warn("--amp requires CUDA; AMP disabled for this run.")
    scaler = GradScaler(device=device.type, enabled=use_amp)

    # ---- TensorBoard ----------------------------------------------------- #
    writer = None
    if _TENSORBOARD:
        writer = SummaryWriter(log_dir=str(out_dir / 'tb'))

    # ---- Training loop --------------------------------------------------- #
    best_dice  = 0.0
    best_epoch = 0
    best_ckpt  = out_dir / 'best.pth'

    t0 = time.time()
    for epoch in range(args.epochs):
        train_stats = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, args.model, use_amp,
        )
        val_stats = validate(
            model, val_loader, criterion, device, args.model, use_amp,
        )
        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']

        # ---- Logging ----------------------------------------------------- #
        if writer is not None:
            writer.add_scalar('train/loss', train_stats['loss'], epoch)
            writer.add_scalar('val/loss',   val_stats['loss'],   epoch)
            writer.add_scalar('val/dice',   val_stats['dice'],   epoch)
            writer.add_scalar('val/iou',    val_stats['iou'],    epoch)
            writer.add_scalar('lr',         current_lr,          epoch)

        # ---- Checkpoint -------------------------------------------------- #
        if val_stats['dice'] > best_dice:
            best_dice  = val_stats['dice']
            best_epoch = epoch + 1
            torch.save(
                {
                    'epoch':      epoch + 1,
                    'model':      args.model,
                    'state_dict': model.state_dict(),
                    'val_dice':   best_dice,
                    'val_iou':    val_stats['iou'],
                    'args':       vars(args),
                },
                best_ckpt,
            )

        # ---- Progress print ---------------------------------------------- #
        if (epoch + 1) % args.print_freq == 0 or epoch == 0:
            elapsed = time.time() - t0
            print(
                f"  epoch {epoch+1:3d}/{args.epochs}  "
                f"lr={current_lr:.2e}  "
                f"train_loss={train_stats['loss']:.4f}  "
                f"val_loss={val_stats['loss']:.4f}  "
                f"val_dice={val_stats['dice']:.4f}  "
                f"val_iou={val_stats['iou']:.4f}  "
                f"[{elapsed:.0f}s]"
            )

    if writer is not None:
        writer.close()

    # ---- Final validation metrics from the best checkpoint --------------- #
    ckpt = torch.load(best_ckpt, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['state_dict'])
    final_val = validate(model, val_loader, criterion, device, args.model, use_amp)

    print(
        f"\n  [fold {fold_idx}] BEST epoch={best_epoch}  "
        f"dice={best_dice:.4f}  iou={final_val['iou']:.4f}  "
        f"sens={final_val['sensitivity']:.4f}  "
        f"spec={final_val['specificity']:.4f}  "
        f"prec={final_val['precision']:.4f}"
    )

    return {
        'fold':          fold_idx,
        'best_epoch':    best_epoch,
        'best_val_dice': best_dice,
        **{f'val_{k}': v for k, v in final_val.items()},
    }


# --------------------------------------------------------------------------- #
# 8.  Data loading helpers                                                    #
# --------------------------------------------------------------------------- #

def _build_phase1_loaders(
    data_root: Path,
    fold_idx:  int,
    args:      argparse.Namespace,
) -> Tuple[DataLoader, DataLoader]:
    """Build Phase 1 (same-scale) train/val loaders for one fold."""
    if args.kfold > 0:
        return build_kfold_loaders(
            data_root=data_root,
            fold_idx=fold_idx,
            n_folds=args.kfold,
            batch_size=args.batch_size,
            img_size=args.img_size,
            seed=args.seed,
            num_workers=args.num_workers,
        )
    # kfold=0 → simple 80/20 holdout
    all_pairs = find_pairs(data_root)
    n_val = max(1, int(len(all_pairs) * 0.2))
    rng   = np.random.RandomState(args.seed)
    perm  = rng.permutation(len(all_pairs)).tolist()
    tr_p  = [all_pairs[i] for i in perm[:-n_val]]
    va_p  = [all_pairs[i] for i in perm[-n_val:]]
    print(f"[simple holdout] train={len(tr_p)}  val={len(va_p)}")
    tr_ds = EGCPhase1Dataset(tr_p, img_size=args.img_size, augment=True)
    va_ds = EGCPhase1Dataset(va_p, img_size=args.img_size, augment=False)
    kw = dict(num_workers=args.num_workers, pin_memory=True)
    return (
        DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                   drop_last=len(tr_ds) >= args.batch_size, **kw),
        DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, **kw),
    )


def _build_loaders(
    data_root: Path,
    fold_idx:  int,
    args:      argparse.Namespace,
) -> Tuple[DataLoader, DataLoader]:
    """Dispatch to the Phase 1 or Phase 2 dataloader factory."""
    if args.phase == 1:
        return _build_phase1_loaders(data_root, fold_idx, args)

    # Phase 2: cross-scale (WLI distant + NBI closeup, separate masks)
    try:
        from data.dataloader_crossscale import build_kfold_loaders as build_cs
        return build_cs(
            data_root=data_root,
            fold_idx=fold_idx,
            n_folds=max(1, args.kfold),
            batch_size=args.batch_size,
            img_size=args.img_size,
            seed=args.seed,
            num_workers=args.num_workers,
        )
    except ImportError:
        print(
            "ERROR: Phase 2 dataloader not yet implemented.\n"
            "       Run with --phase 1 for now.",
            file=sys.stderr,
        )
        sys.exit(1)


# --------------------------------------------------------------------------- #
# 9.  Summary                                                                 #
# --------------------------------------------------------------------------- #

def _print_summary(fold_results: List[Dict]) -> None:
    """Print cross-fold mean ± std for the key validation metrics."""
    metrics = ('best_val_dice', 'val_iou', 'val_sensitivity',
               'val_specificity', 'val_precision')
    labels  = ('Dice', 'IoU', 'Sens', 'Spec', 'Prec')

    print(f"\n{'='*60}")
    print(f"CROSS-FOLD SUMMARY  ({len(fold_results)} folds)")
    print(f"{'='*60}")
    for key, label in zip(metrics, labels):
        vals = [r[key] for r in fold_results]
        print(f"  {label:5s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    print(f"{'='*60}\n")


# --------------------------------------------------------------------------- #
# 10.  CLI                                                                    #
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    """Parse all CLI arguments (see CLAUDE.md §Training Script)."""
    p = argparse.ArgumentParser(
        description='DSCMFNet training with K-fold cross-validation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    p.add_argument('--data_root',    default='processed_data')
    p.add_argument('--img_size',     type=int,   default=352)
    p.add_argument('--phase',        type=int,   default=1,  choices=[1, 2])

    # Training
    p.add_argument('--epochs',       type=int,   default=200)
    p.add_argument('--batch_size',   type=int,   default=4)
    p.add_argument('--lr',           type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--freeze_stages',type=int,   default=2)
    p.add_argument('--lambda_bdy',   type=float, default=0.5)
    p.add_argument('--alpha_ds',     type=float, default=0.3)
    p.add_argument('--amp',          action='store_true',
                   help='Enable mixed-precision training (requires CUDA)')

    # CV
    p.add_argument('--kfold',        type=int,   default=5,
                   help='K-fold splits; 0 = simple 80/20 holdout')
    p.add_argument('--seed',         type=int,   default=42)

    # System
    p.add_argument('--num_workers',  type=int,   default=0)
    p.add_argument('--device',       default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output',       default='./output')
    p.add_argument('--print_freq',   type=int,   default=10)

    # Model
    p.add_argument('--model',        default='dscmfnet',
                   choices=['nbi_only', 'wli_only', 'early_fusion', 'dscmfnet'])
    p.add_argument('--fusion_mode',  default='concat',
                   choices=['concat', 'cmfim', 'cmfim_no_sam', 'cmfim_no_boundary'],
                   help='DSCMFNet fusion strategy (ignored for non-dscmfnet models)')
    p.add_argument('--pretrained',   action=argparse.BooleanOptionalAction,
                   default=True,
                   help='Use ImageNet-pretrained encoder weights')

    return p.parse_args()


# --------------------------------------------------------------------------- #
# 11.  Main                                                                   #
# --------------------------------------------------------------------------- #

def main() -> None:
    """Entry point: parse args, run K-fold training, print summary."""
    args = parse_args()
    seed_everything(args.seed)

    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"ERROR: --data_root '{data_root}' not found.", file=sys.stderr)
        sys.exit(1)

    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    n_folds = max(1, args.kfold)
    if not _TENSORBOARD:
        print("[WARN] tensorboard not installed; install with `pip install tensorboard`")

    print(f"\nDSCMFNet training")
    print(f"  model       : {args.model}")
    print(f"  fusion_mode : {args.fusion_mode}")
    print(f"  phase       : {args.phase}")
    print(f"  device      : {args.device}")
    print(f"  epochs      : {args.epochs}")
    print(f"  batch_size  : {args.batch_size}")
    print(f"  lr          : {args.lr}")
    print(f"  freeze      : {args.freeze_stages} stages")
    print(f"  kfold       : {args.kfold} ({'simple holdout' if args.kfold == 0 else 'K-fold CV'})")
    print(f"  pretrained  : {args.pretrained}")
    print(f"  output      : {out_root.resolve()}\n")

    fold_results: List[Dict] = []

    if args.kfold == 0:
        # Simple holdout — single "fold"
        train_loader, val_loader = _build_loaders(data_root, 0, args)
        result = train_fold(0, train_loader, val_loader, args)
        fold_results.append(result)
    else:
        for fold_idx in range(n_folds):
            print(f"\n{'─'*60}")
            print(f"  Fold {fold_idx + 1} / {n_folds}")
            print(f"{'─'*60}")
            train_loader, val_loader = _build_loaders(data_root, fold_idx, args)
            result = train_fold(fold_idx, train_loader, val_loader, args)
            fold_results.append(result)

    _print_summary(fold_results)


if __name__ == '__main__':
    main()
