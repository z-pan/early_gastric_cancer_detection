"""models/dscmfnet.py — DSCMFNet Phase 1: Dual-Stream + Concat Fusion.

This module implements the Phase 1 variant of the Dual-Stream Cross-Modal
Fusion Network (DSCMFNet) for pixel-level segmentation of Early Gastric
Cancer (EGC) from paired WLI + NBI endoscopic images.

Phase 1 fusion strategy (ablation #4 in CLAUDE.md):
  * Both streams share the same Progressive Decoder as the single-stream
    baselines (``models/single_stream.py``).
  * At every spatial scale, each stream is projected to a unified channel
    dimension, concatenated, and reduced with a 1×1 Conv.
  * **CMFIM, Spatial Alignment Module, and Bidirectional Cross-Attention are
    NOT implemented here** — they will replace ``ConcatFusionBlock`` in
    Phase 2 (STEP 7) while keeping the ``forward(wli, nbi)`` interface intact.

Architecture::

    WLI  ──► PVTv2-B2  ──► [w1(64),  w2(128), w3(320), w4(512)]
                                │        │        │        │
                             Fuse1   Fuse2    Fuse3    Fuse4      (concat fusion)
                                │        │        │        │
    NBI  ──► ResNet34  ──► [n1(64),  n2(128), n3(256), n4(512)]
                                │        │        │        │
                             [F1(64), F2(128), F3(256), F4(512)]  fused features
                                                │
                                    ProgressiveDecoder
                                                │
                                ┌───────────────┴────────────────┐
                           SegHead (1ch)               BoundaryHead (1ch)
                                + DeepSupervision ×3 (H/4, H/8, H/16)

Output::

    (seg_pred, bdy_pred, [ds_h4, ds_h8, ds_h16])

The ``forward(wli, nbi)`` signature is frozen; Phase 2 will only swap out
the ``self.fusion_*`` sub-modules.
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.losses import DSCMFNetLoss
from models.single_stream import (
    ProgressiveDecoder,
    _PVTv2B2Encoder,
    _ResNet34Encoder,
)

# --------------------------------------------------------------------------- #
# Fused channel dimensions at each spatial scale                              #
# --------------------------------------------------------------------------- #
# H/4  → 64d,  H/8 → 128d,  H/16 → 256d,  H/32 → 512d
# These are the skip-connection channels seen by ProgressiveDecoder.
_FUSED_CHANNELS: Tuple[int, int, int, int] = (64, 128, 256, 512)


# --------------------------------------------------------------------------- #
# Concat Fusion Block (Phase 1)                                               #
# --------------------------------------------------------------------------- #

class ConcatFusionBlock(nn.Module):
    """Phase-1 scale-wise fusion: project → concatenate → reduce.

    Each stream is independently projected to ``out_c`` channels via
    ``1×1 Conv → BN → GELU``, then the two projections are concatenated
    and reduced back to ``out_c`` with a final 1×1 Conv.

    This module is designed to be **drop-in replaceable** by a full CMFIM
    block (Phase 2) with the same ``(wli_feat, nbi_feat) → fused`` signature.

    Args:
        wli_c: WLI encoder output channels at this scale.
        nbi_c: NBI encoder output channels at this scale.
        out_c: Unified fused output channels.
    """

    def __init__(self, wli_c: int, nbi_c: int, out_c: int) -> None:
        super().__init__()
        self.proj_wli = nn.Sequential(
            nn.Conv2d(wli_c, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.GELU(),
        )
        self.proj_nbi = nn.Sequential(
            nn.Conv2d(nbi_c, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.GELU(),
        )
        self.fuse = nn.Conv2d(out_c * 2, out_c, kernel_size=1, bias=False)

    def forward(
        self, wli_feat: torch.Tensor, nbi_feat: torch.Tensor
    ) -> torch.Tensor:
        """Fuse one spatial scale.

        Args:
            wli_feat: WLI feature map ``(B, wli_c, H, W)``.
            nbi_feat: NBI feature map ``(B, nbi_c, H, W)``.

        Returns:
            Fused feature map ``(B, out_c, H, W)``.
        """
        w = self.proj_wli(wli_feat)
        n = self.proj_nbi(nbi_feat)
        return self.fuse(torch.cat([w, n], dim=1))


# --------------------------------------------------------------------------- #
# DSCMFNet                                                                    #
# --------------------------------------------------------------------------- #

class DSCMFNet(nn.Module):
    """DSCMFNet Phase 1 — Dual-Stream Concat-Fusion Segmentation Network.

    Accepts a paired (WLI, NBI) input and returns multi-output predictions
    compatible with :class:`~models.losses.DSCMFNetLoss`.

    The ``forward(wli, nbi)`` interface is intentionally fixed so that
    Phase 2 CMFIM fusion modules can be swapped in without touching any
    training or evaluation code.

    Args:
        pretrained:     Load ImageNet-pretrained encoder weights.

    Attributes:
        wli_encoder:    PVTv2-B2 encoder (Stream A — WLI context).
        nbi_encoder:    ResNet34 encoder (Stream B — NBI detail).
        fusion_{1..4}:  :class:`ConcatFusionBlock` at H/4 … H/32.
        decoder:        :class:`~models.single_stream.ProgressiveDecoder`.
    """

    #: WLI encoder output channels: [C1, C2, C3, C4]
    _WLI_CH: Tuple[int, int, int, int] = _PVTv2B2Encoder.OUT_CHANNELS    # (64, 128, 320, 512)
    #: NBI encoder output channels: [C1, C2, C3, C4]
    _NBI_CH: Tuple[int, int, int, int] = _ResNet34Encoder.OUT_CHANNELS   # (64, 128, 256, 512)
    #: Fused feature channels passed to the decoder
    _FUSED_CH: Tuple[int, int, int, int] = _FUSED_CHANNELS               # (64, 128, 256, 512)

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()

        # ---- Encoders ----------------------------------------------------- #
        self.wli_encoder = _PVTv2B2Encoder(pretrained=pretrained, in_chans=3)
        self.nbi_encoder = _ResNet34Encoder(pretrained=pretrained)

        # ---- Fusion blocks (one per spatial scale) ------------------------ #
        w = self._WLI_CH
        n = self._NBI_CH
        f = self._FUSED_CH

        self.fusion_1 = ConcatFusionBlock(w[0], n[0], f[0])  # H/4
        self.fusion_2 = ConcatFusionBlock(w[1], n[1], f[1])  # H/8
        self.fusion_3 = ConcatFusionBlock(w[2], n[2], f[2])  # H/16
        self.fusion_4 = ConcatFusionBlock(w[3], n[3], f[3])  # H/32

        # ---- Shared Progressive Decoder ----------------------------------- #
        self.decoder = ProgressiveDecoder(self._FUSED_CH)

    # ---------------------------------------------------------------------- #
    # Forward                                                                 #
    # ---------------------------------------------------------------------- #

    def forward(
        self,
        wli: torch.Tensor,
        nbi: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Run a paired (WLI, NBI) forward pass.

        This signature is **frozen** — Phase 2 will only modify the
        internal fusion modules while keeping this interface unchanged.

        Args:
            wli: ``(B, 3, H, W)`` WLI image (ImageNet-normalised).
            nbi: ``(B, 3, H, W)`` NBI image (ImageNet-normalised).

        Returns:
            Tuple of:

            * ``seg_pred``  — ``(B, 1, H, W)`` segmentation logits.
            * ``bdy_pred``  — ``(B, 1, H, W)`` boundary logits.
            * ``ds_preds``  — ``[ds_h4, ds_h8, ds_h16]`` auxiliary logits
              at H/4, H/8, H/16 (un-upsampled; loss handles resize).
        """
        # ---- Dual-stream encoding ---------------------------------------- #
        w1, w2, w3, w4 = self.wli_encoder(wli)
        n1, n2, n3, n4 = self.nbi_encoder(nbi)

        # ---- Scale-wise fusion ------------------------------------------- #
        f1 = self.fusion_1(w1, n1)   # (B,  64, H/4,  W/4)
        f2 = self.fusion_2(w2, n2)   # (B, 128, H/8,  W/8)
        f3 = self.fusion_3(w3, n3)   # (B, 256, H/16, W/16)
        f4 = self.fusion_4(w4, n4)   # (B, 512, H/32, W/32)

        # ---- Progressive decoding ---------------------------------------- #
        return self.decoder(f1, f2, f3, f4)

    # ---------------------------------------------------------------------- #
    # Encoder freezing                                                        #
    # ---------------------------------------------------------------------- #

    def freeze_encoder_stages(self, num_stages: int) -> None:
        """Freeze the first ``num_stages`` stages of both encoders.

        Freezing early encoder stages reduces GPU memory and prevents
        over-fitting when fine-tuning on small EGC datasets (47 cases).

        **PVTv2-B2 stage mapping** (timm ``FeatureListNet`` layout):

        ====  ============================================
        idx   frozen sub-modules
        ====  ============================================
        0     ``patch_embed`` (7×7 overlap embed, stride 4) + ``stages_0``
        1     ``stages_1`` (includes its ``downsample`` embed)
        2     ``stages_2``
        3     ``stages_3``
        ====  ============================================

        **ResNet34 stage mapping**:

        ====  ==========================================
        idx   frozen sub-modules
        ====  ==========================================
        0     ``stem`` (conv1 + bn1 + relu + maxpool) + ``layer1``
        1     ``layer2``
        2     ``layer3``
        3     ``layer4``
        ====  ==========================================

        Args:
            num_stages: Number of leading encoder stages to freeze (0–4).
                        0 → nothing frozen; 4 → entire encoder frozen.
        """
        if num_stages == 0:
            return
        num_stages = min(num_stages, 4)

        # ---- Freeze PVTv2-B2 (WLI encoder) ------------------------------- #
        pvt = self.wli_encoder.backbone   # FeatureListNet
        pvt_stage_modules = [
            [pvt.patch_embed, getattr(pvt, 'stages_0')],   # stage 0
            [getattr(pvt, 'stages_1')],                     # stage 1
            [getattr(pvt, 'stages_2')],                     # stage 2
            [getattr(pvt, 'stages_3')],                     # stage 3
        ]
        for stage_idx in range(num_stages):
            for mod in pvt_stage_modules[stage_idx]:
                for p in mod.parameters():
                    p.requires_grad = False

        # ---- Freeze ResNet34 (NBI encoder) ------------------------------- #
        rn = self.nbi_encoder
        rn_stage_modules = [
            [rn.stem, rn.layer1],   # stage 0
            [rn.layer2],             # stage 1
            [rn.layer3],             # stage 2
            [rn.layer4],             # stage 3
        ]
        for stage_idx in range(num_stages):
            for mod in rn_stage_modules[stage_idx]:
                for p in mod.parameters():
                    p.requires_grad = False

    def unfreeze_all(self) -> None:
        """Re-enable gradient computation for all parameters."""
        for p in self.parameters():
            p.requires_grad = True


# --------------------------------------------------------------------------- #
# Parameter counting utility                                                  #
# --------------------------------------------------------------------------- #

def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count total and trainable parameters in a model.

    Args:
        model: Any :class:`torch.nn.Module`.

    Returns:
        Dict with keys ``'total'`` and ``'trainable'`` (integer counts).
    """
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total': total, 'trainable': trainable}


# --------------------------------------------------------------------------- #
# Smoke test                                                                  #
# --------------------------------------------------------------------------- #

def _smoke_test() -> None:
    """Run a dummy forward pass and print parameter counts + output shapes."""

    device = torch.device('cpu')
    B, H, W = 2, 352, 352
    wli = torch.randn(B, 3, H, W, device=device)
    nbi = torch.randn(B, 3, H, W, device=device)
    mask = torch.zeros(B, 1, H, W, device=device)
    mask[:, :, 80:200, 100:250] = 1.0

    print(f"\n{'='*65}")
    print("DSCMFNet Phase 1 smoke test  (pretrained=False)")
    print(f"{'='*65}\n")

    model = DSCMFNet(pretrained=False).eval().to(device)
    criterion = DSCMFNetLoss()

    # ---- Parameter counts ------------------------------------------------ #
    params = count_parameters(model)
    print(f"  Parameters (full):")
    print(f"    Total     : {params['total']:,}")
    print(f"    Trainable : {params['trainable']:,}")

    # ---- Per-section breakdown ------------------------------------------- #
    sections = {
        'wli_encoder' : model.wli_encoder,
        'nbi_encoder' : model.nbi_encoder,
        'fusion_1..4' : nn.ModuleList([
            model.fusion_1, model.fusion_2,
            model.fusion_3, model.fusion_4,
        ]),
        'decoder'     : model.decoder,
    }
    print(f"\n  Per-section parameter counts:")
    for name, mod in sections.items():
        n = sum(p.numel() for p in mod.parameters())
        print(f"    {name:16s}: {n:>12,}")

    # ---- Forward pass ---------------------------------------------------- #
    with torch.no_grad():
        seg, bdy, ds = model(wli, nbi)

    loss = criterion(seg, bdy, ds, mask)

    print(f"\n  Output shapes:")
    print(f"    seg_pred : {tuple(seg.shape)}")
    print(f"    bdy_pred : {tuple(bdy.shape)}")
    for i, d in enumerate(ds):
        tag = ['H/4', 'H/8', 'H/16'][i]
        print(f"    ds[{i}] {tag:4s}: {tuple(d.shape)}")
    print(f"    loss     : {loss.item():.4f}")

    # ---- Freeze test ----------------------------------------------------- #
    print(f"\n  freeze_encoder_stages(2) …")
    model.train()
    model.freeze_encoder_stages(2)
    params_frozen = count_parameters(model)
    print(f"    Trainable after freeze: {params_frozen['trainable']:,}")
    frozen_n = params['trainable'] - params_frozen['trainable']
    print(f"    Frozen parameters     : {frozen_n:,}")

    model.unfreeze_all()
    params_unfrozen = count_parameters(model)
    print(f"    Trainable after unfreeze: {params_unfrozen['trainable']:,}")
    assert params_unfrozen['trainable'] == params['total']

    print(f"\nSmoke test PASSED.")
    print("=" * 65)


if __name__ == '__main__':
    _smoke_test()
