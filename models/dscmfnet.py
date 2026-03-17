"""models/dscmfnet.py — DSCMFNet: Dual-Stream Cross-Modal Fusion Network.

Pixel-level segmentation of Early Gastric Cancer (EGC) from paired
WLI + NBI endoscopic images.

Fusion modes (selected via ``fusion_mode`` constructor arg):

* ``'concat'``           — Phase 1: all scales use :class:`ConcatFusionBlock`
                            (original Phase-1 logic, unchanged).
* ``'cmfim'``            — Phase 2: scale 1/4 uses concat, scales 1/8 · 1/16 · 1/32
                            use full :class:`CMFIM` (SAM + bidirectional attention).
* ``'cmfim_no_sam'``     — Ablation #6: same as ``'cmfim'`` but SAM is bypassed.
* ``'cmfim_no_boundary'``— Ablation #7: same as ``'cmfim'`` but the boundary head
                            is disabled; ``forward()`` returns ``bdy_pred = None``.

Architecture::

    WLI  ──► PVTv2-B2  ──► [w1(64),  w2(128), w3(320), w4(512)]
                                │        │        │        │
                             Concat   Fuse2    Fuse3    Fuse4    ← mode-dependent
                                │        │        │        │
    NBI  ──► ResNet34  ──► [n1(64),  n2(128), n3(256), n4(512)]
                                │        │        │        │
                             [F1(64), F2(128), F3(256), F4(512)]  fused features
                                                │
                                    ProgressiveDecoder
                                                │
                                ┌───────────────┴────────────────┐
                           SegHead (1ch)               BoundaryHead (1ch)*
                                + DeepSupervision ×3

Output::

    (seg_pred, bdy_pred, [ds_h4, ds_h8, ds_h16])

    * bdy_pred is ``None`` when ``fusion_mode='cmfim_no_boundary'``.
"""

from typing import Dict, List, Optional, Tuple

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
# Constants                                                                   #
# --------------------------------------------------------------------------- #

# H/4→64, H/8→128, H/16→256, H/32→512  (skip-connection channels for decoder)
_FUSED_CHANNELS: Tuple[int, int, int, int] = (64, 128, 256, 512)

_VALID_FUSION_MODES = frozenset({
    'concat',
    'cmfim',
    'cmfim_no_sam',
    'cmfim_no_boundary',
})


# --------------------------------------------------------------------------- #
# Phase-1: Concat Fusion Block                                                #
# --------------------------------------------------------------------------- #

class ConcatFusionBlock(nn.Module):
    """Phase-1 scale-wise fusion: project → concatenate → reduce.

    Each stream is independently projected to ``out_c`` channels via
    ``1×1 Conv → BN → GELU``, then the two projections are concatenated
    and reduced back to ``out_c`` with a final 1×1 Conv.

    Drop-in replaceable by :class:`CMFIM` — identical
    ``(wli_feat, nbi_feat) → fused`` call signature.

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
            wli_feat: ``(B, wli_c, H, W)``
            nbi_feat: ``(B, nbi_c, H, W)``

        Returns:
            Fused feature ``(B, out_c, H, W)``.
        """
        w = self.proj_wli(wli_feat)
        n = self.proj_nbi(nbi_feat)
        return self.fuse(torch.cat([w, n], dim=1))


# --------------------------------------------------------------------------- #
# Phase-2 building block 1: Spatial Alignment Module (SAM)                   #
# --------------------------------------------------------------------------- #

class SpatialAlignmentModule(nn.Module):
    """Predict a 2-D deformation field and warp NBI features into WLI space.

    The offset network maps ``concat(nbi, wli)`` → 2-channel offset field
    ``(dx, dy)`` in normalised ``[-1, 1]`` grid coordinates.  Weights are
    zero-initialised so the warp starts as the identity transform; the offset
    is additionally scaled by 0.1 for training stability.

    Args:
        dim: Unified feature channels (both streams projected to this before SAM).
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.offset_net = nn.Sequential(
            # concat(nbi, wli): 2*dim → dim
            nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            # dim → 2-channel offset field
            nn.Conv2d(dim, 2, kernel_size=3, padding=1, bias=True),
        )
        # Zero-init: identity warp at the start of training
        nn.init.zeros_(self.offset_net[-1].weight)
        nn.init.zeros_(self.offset_net[-1].bias)

    def forward(
        self,
        nbi_feat: torch.Tensor,
        wli_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Warp ``nbi_feat`` to align with WLI coordinate space.

        Args:
            nbi_feat: NBI projected features ``(B, C, H, W)``.
            wli_feat: WLI projected features ``(B, C, H, W)``.

        Returns:
            Spatially aligned NBI features ``(B, C, H, W)``.
        """
        B, _, H, W = nbi_feat.shape

        # Predict 2-D offset field  (B, 2, H, W), scaled for stability
        offset = self.offset_net(torch.cat([nbi_feat, wli_feat], dim=1)) * 0.1

        # Build normalised base sampling grid  (B, H, W, 2) in [-1, 1]
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, dtype=nbi_feat.dtype, device=nbi_feat.device),
            torch.linspace(-1, 1, W, dtype=nbi_feat.dtype, device=nbi_feat.device),
            indexing='ij',
        )
        base_grid = torch.stack([grid_x, grid_y], dim=-1)          # (H, W, 2)
        base_grid = base_grid.unsqueeze(0).expand(B, -1, -1, -1)   # (B, H, W, 2)

        # Displace grid: permute offset from (B, 2, H, W) → (B, H, W, 2)
        sampling_grid = base_grid + offset.permute(0, 2, 3, 1)

        return F.grid_sample(
            nbi_feat, sampling_grid,
            mode='bilinear', padding_mode='border', align_corners=True,
        )


# --------------------------------------------------------------------------- #
# Phase-2 building block 2: Bidirectional Cross-Modal Attention               #
# --------------------------------------------------------------------------- #

class CrossModalAttention(nn.Module):
    """Bidirectional cross-modal attention between WLI and aligned NBI features.

    Two cross-attention directions run in parallel:

    * **Dir-1** ``Q=WLI, K/V=NBI``:   WLI features absorb NBI micro-texture.
    * **Dir-2** ``Q=NBI, K/V=WLI``:   NBI features absorb WLI global context.

    Standard dot-product attention over flattened spatial positions.
    Query / Key channels are compressed by ``reduction`` to limit memory;
    for small datasets this single-head formulation suffices.
    Each direction adds a residual connection followed by BatchNorm.

    Args:
        dim:       Unified feature channels.
        reduction: Q/K channel compression factor (default 4).
    """

    def __init__(self, dim: int, reduction: int = 4) -> None:
        super().__init__()
        rdim = max(dim // reduction, 16)   # reduced dim for Q, K
        self._scale = rdim ** -0.5

        # Direction 1: Q=WLI, K/V=NBI  →  enhanced WLI
        self.q_wli  = nn.Conv2d(dim, rdim, kernel_size=1, bias=False)
        self.k_nbi  = nn.Conv2d(dim, rdim, kernel_size=1, bias=False)
        self.v_nbi  = nn.Conv2d(dim, dim,  kernel_size=1, bias=False)
        self.bn_wli = nn.BatchNorm2d(dim)

        # Direction 2: Q=NBI, K/V=WLI  →  enhanced NBI
        self.q_nbi  = nn.Conv2d(dim, rdim, kernel_size=1, bias=False)
        self.k_wli  = nn.Conv2d(dim, rdim, kernel_size=1, bias=False)
        self.v_wli  = nn.Conv2d(dim, dim,  kernel_size=1, bias=False)
        self.bn_nbi = nn.BatchNorm2d(dim)

    def _dot_attn(
        self,
        q: torch.Tensor,   # (B, rdim, H, W)
        k: torch.Tensor,   # (B, rdim, H, W)
        v: torch.Tensor,   # (B, dim,  H, W)
    ) -> torch.Tensor:
        """Single-direction spatial dot-product attention.

        Returns:
            Context tensor ``(B, dim, H, W)``.
        """
        B, C_v, H, W = v.shape
        q_flat = q.flatten(2).transpose(1, 2)   # (B, HW, rdim)
        k_flat = k.flatten(2)                    # (B, rdim, HW)
        v_flat = v.flatten(2).transpose(1, 2)    # (B, HW, C_v)

        attn = torch.softmax(q_flat @ k_flat * self._scale, dim=-1)  # (B, HW, HW)
        out  = attn @ v_flat                                           # (B, HW, C_v)
        return out.transpose(1, 2).reshape(B, C_v, H, W)

    def forward(
        self,
        wli_feat:    torch.Tensor,
        nbi_aligned: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply bidirectional cross-modal attention.

        Args:
            wli_feat:    WLI projected features ``(B, dim, H, W)``.
            nbi_aligned: NBI projected (+ optionally SAM-warped) features
                         ``(B, dim, H, W)``.

        Returns:
            ``(enhanced_wli, enhanced_nbi)`` each ``(B, dim, H, W)``.
        """
        # Dir-1: WLI absorbs NBI texture
        enh_wli = self.bn_wli(
            wli_feat + self._dot_attn(
                self.q_wli(wli_feat),
                self.k_nbi(nbi_aligned),
                self.v_nbi(nbi_aligned),
            )
        )

        # Dir-2: NBI absorbs WLI context
        enh_nbi = self.bn_nbi(
            nbi_aligned + self._dot_attn(
                self.q_nbi(nbi_aligned),
                self.k_wli(wli_feat),
                self.v_wli(wli_feat),
            )
        )

        return enh_wli, enh_nbi


# --------------------------------------------------------------------------- #
# Phase-2 composite block: CMFIM                                              #
# --------------------------------------------------------------------------- #

class CMFIM(nn.Module):
    """Cross-Modal Feature Integration Module (CMFIM).

    Full fusion pipeline for one spatial scale:

    1. **Channel Projection** — 1×1 Conv + BN + GELU per stream → unified dim.
    2. **SAM** (optional) — warp NBI features toward WLI coordinate space.
    3. **Bidirectional CrossModalAttention** — WLI ↔ NBI context exchange.
    4. **Feature Fusion** — concat → 1×1 Conv → 3×3 Conv + BN + GELU → F_i.

    Used at 1/8, 1/16, 1/32 scales; 1/4 scale always uses :class:`ConcatFusionBlock`
    (too large for full attention).  Drop-in replacement for :class:`ConcatFusionBlock`.

    Args:
        wli_c:   WLI encoder channels at this scale.
        nbi_c:   NBI encoder channels at this scale.
        out_c:   Unified output channels.
        use_sam: Enable Spatial Alignment Module (set ``False`` for ablation #6).
    """

    def __init__(
        self,
        wli_c:   int,
        nbi_c:   int,
        out_c:   int,
        use_sam: bool = True,
    ) -> None:
        super().__init__()

        # Step 1 — Channel projection
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

        # Step 2 — Spatial Alignment Module (optional)
        self.use_sam = use_sam
        self.sam = SpatialAlignmentModule(out_c) if use_sam else None

        # Step 3 — Bidirectional cross-modal attention
        self.cross_attn = CrossModalAttention(out_c, reduction=4)

        # Step 4 — Feature fusion
        self.fuse = nn.Sequential(
            nn.Conv2d(out_c * 2, out_c, kernel_size=1, bias=False),
            nn.Conv2d(out_c, out_c,     kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.GELU(),
        )

    def forward(
        self,
        wli_feat: torch.Tensor,
        nbi_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse one spatial scale with the full CMFIM pipeline.

        Args:
            wli_feat: ``(B, wli_c, H, W)``
            nbi_feat: ``(B, nbi_c, H, W)``

        Returns:
            ``(B, out_c, H, W)``
        """
        w = self.proj_wli(wli_feat)   # (B, out_c, H, W)
        n = self.proj_nbi(nbi_feat)   # (B, out_c, H, W)

        if self.use_sam:
            n = self.sam(n, w)        # warp NBI → WLI space

        enh_w, enh_n = self.cross_attn(w, n)
        return self.fuse(torch.cat([enh_w, enh_n], dim=1))


# --------------------------------------------------------------------------- #
# DSCMFNet                                                                    #
# --------------------------------------------------------------------------- #

class DSCMFNet(nn.Module):
    """DSCMFNet — Dual-Stream Cross-Modal Fusion Segmentation Network.

    Accepts paired (WLI, NBI) inputs and returns multi-output predictions
    compatible with :class:`~models.losses.DSCMFNetLoss`.

    The fusion strategy is selected via ``fusion_mode``:

    ========================  ========================================================
    ``fusion_mode``           Fusion at each spatial scale
    ========================  ========================================================
    ``'concat'``              All 4 scales → :class:`ConcatFusionBlock`
    ``'cmfim'``               1/4 → concat;  1/8 · 1/16 · 1/32 → :class:`CMFIM`
    ``'cmfim_no_sam'``        Same as ``'cmfim'`` but SAM step skipped (ablation)
    ``'cmfim_no_boundary'``   Same as ``'cmfim'`` but boundary head disabled;
                              ``forward()`` returns ``bdy_pred = None``
    ========================  ========================================================

    The ``forward(wli, nbi)`` interface is frozen across all modes so that
    training and evaluation code never needs to branch on ``fusion_mode``.

    Args:
        pretrained:   Load ImageNet-pretrained encoder weights.
        fusion_mode:  One of the four values listed above (default ``'concat'``).
    """

    _WLI_CH:   Tuple[int, int, int, int] = _PVTv2B2Encoder.OUT_CHANNELS  # (64,128,320,512)
    _NBI_CH:   Tuple[int, int, int, int] = _ResNet34Encoder.OUT_CHANNELS  # (64,128,256,512)
    _FUSED_CH: Tuple[int, int, int, int] = _FUSED_CHANNELS                # (64,128,256,512)

    def __init__(
        self,
        pretrained:  bool = True,
        fusion_mode: str  = 'concat',
    ) -> None:
        super().__init__()

        if fusion_mode not in _VALID_FUSION_MODES:
            raise ValueError(
                f"fusion_mode={fusion_mode!r} is not valid. "
                f"Choose from {sorted(_VALID_FUSION_MODES)}."
            )
        self.fusion_mode = fusion_mode

        # ---- Encoders ------------------------------------------------------- #
        self.wli_encoder = _PVTv2B2Encoder(pretrained=pretrained, in_chans=3)
        self.nbi_encoder = _ResNet34Encoder(pretrained=pretrained)

        w, n, f = self._WLI_CH, self._NBI_CH, self._FUSED_CH

        # ---- Fusion blocks -------------------------------------------------- #
        # Scale 1 (H/4): always ConcatFusionBlock — CMFIM attention at 88×88
        # would be (88×88)² = 59M elements per sample; prohibitively expensive.
        self.fusion_1 = ConcatFusionBlock(w[0], n[0], f[0])

        if fusion_mode == 'concat':
            # All scales: simple concat (Phase 1, unchanged)
            self.fusion_2 = ConcatFusionBlock(w[1], n[1], f[1])
            self.fusion_3 = ConcatFusionBlock(w[2], n[2], f[2])
            self.fusion_4 = ConcatFusionBlock(w[3], n[3], f[3])
        else:
            # Scales 2–4: full CMFIM (with or without SAM)
            use_sam = (fusion_mode != 'cmfim_no_sam')
            self.fusion_2 = CMFIM(w[1], n[1], f[1], use_sam=use_sam)  # H/8
            self.fusion_3 = CMFIM(w[2], n[2], f[2], use_sam=use_sam)  # H/16
            self.fusion_4 = CMFIM(w[3], n[3], f[3], use_sam=use_sam)  # H/32

        # ---- Progressive Decoder -------------------------------------------- #
        self.decoder = ProgressiveDecoder(self._FUSED_CH)

    # ---------------------------------------------------------------------- #
    # Forward                                                                 #
    # ---------------------------------------------------------------------- #

    def forward(
        self,
        wli: torch.Tensor,
        nbi: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[torch.Tensor]]:
        """Run a paired (WLI, NBI) forward pass.

        The call signature is **frozen** across all ``fusion_mode`` values.

        Args:
            wli: ``(B, 3, H, W)`` WLI image (ImageNet-normalised).
            nbi: ``(B, 3, H, W)`` NBI image (ImageNet-normalised).

        Returns:
            Tuple of:

            * ``seg_pred`` — ``(B, 1, H, W)`` segmentation logits.
            * ``bdy_pred`` — ``(B, 1, H, W)`` boundary logits, or ``None``
              when ``fusion_mode='cmfim_no_boundary'``.
            * ``ds_preds`` — ``[ds_h4, ds_h8, ds_h16]`` auxiliary logits.
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
        seg, bdy, ds = self.decoder(f1, f2, f3, f4)

        # 'cmfim_no_boundary': disable boundary head for this ablation
        if self.fusion_mode == 'cmfim_no_boundary':
            bdy = None

        return seg, bdy, ds

    # ---------------------------------------------------------------------- #
    # Utilities                                                               #
    # ---------------------------------------------------------------------- #

    def freeze_encoder_stages(self, num_stages: int) -> None:
        """Freeze the first ``num_stages`` stages of both encoders.

        **PVTv2-B2 stage mapping** (timm ``FeatureListNet`` layout):

        ====  ============================================
        idx   frozen sub-modules
        ====  ============================================
        0     ``patch_embed`` + ``stages_0``
        1     ``stages_1``
        2     ``stages_2``
        3     ``stages_3``
        ====  ============================================

        **ResNet34 stage mapping**:

        ====  ==========================================
        idx   frozen sub-modules
        ====  ==========================================
        0     ``stem`` (conv1+bn1+relu+maxpool) + ``layer1``
        1     ``layer2``
        2     ``layer3``
        3     ``layer4``
        ====  ==========================================

        Args:
            num_stages: Number of leading stages to freeze (0–4).
        """
        if num_stages == 0:
            return
        num_stages = min(num_stages, 4)

        pvt = self.wli_encoder.backbone
        pvt_groups = [
            [pvt.patch_embed, getattr(pvt, 'stages_0')],
            [getattr(pvt, 'stages_1')],
            [getattr(pvt, 'stages_2')],
            [getattr(pvt, 'stages_3')],
        ]
        for i in range(num_stages):
            for mod in pvt_groups[i]:
                for p in mod.parameters():
                    p.requires_grad = False

        rn = self.nbi_encoder
        rn_groups = [
            [rn.stem, rn.layer1],
            [rn.layer2],
            [rn.layer3],
            [rn.layer4],
        ]
        for i in range(num_stages):
            for mod in rn_groups[i]:
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
        Dict with keys ``'total'`` and ``'trainable'``.
    """
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total': total, 'trainable': trainable}


def _fusion_param_count(model: DSCMFNet) -> int:
    """Sum parameters across all four fusion_* sub-modules."""
    return sum(
        sum(p.numel() for p in getattr(model, f'fusion_{i}').parameters())
        for i in range(1, 5)
    )


# --------------------------------------------------------------------------- #
# Smoke test                                                                  #
# --------------------------------------------------------------------------- #

def _smoke_test() -> None:
    """Forward-pass all fusion modes and print parameter comparison."""

    device = torch.device('cpu')
    B, H, W = 2, 352, 352
    wli  = torch.randn(B, 3, H, W, device=device)
    nbi  = torch.randn(B, 3, H, W, device=device)
    mask = torch.zeros(B, 1, H, W, device=device)
    mask[:, :, 80:200, 100:250] = 1.0

    criterion = DSCMFNetLoss()

    print(f"\n{'='*65}")
    print("DSCMFNet smoke test — all fusion modes  (pretrained=False)")
    print(f"{'='*65}\n")

    # ---- Forward pass for every mode ------------------------------------ #
    modes = ['concat', 'cmfim', 'cmfim_no_sam', 'cmfim_no_boundary']
    for mode in modes:
        model = DSCMFNet(pretrained=False, fusion_mode=mode).eval().to(device)
        with torch.no_grad():
            seg, bdy, ds = model(wli, nbi)
        loss = criterion(seg, bdy, ds, mask)

        bdy_info = tuple(bdy.shape) if bdy is not None else 'None (disabled)'
        print(f"  fusion_mode='{mode}'")
        print(f"    seg  : {tuple(seg.shape)}")
        print(f"    bdy  : {bdy_info}")
        print(f"    ds   : {[tuple(d.shape) for d in ds]}")
        print(f"    loss : {loss.item():.4f}")
        print()

    # ---- Parameter comparison: concat vs cmfim -------------------------- #
    print(f"{'─'*65}")
    print("  Parameter delta  concat → cmfim")
    print(f"{'─'*65}")

    m_concat = DSCMFNet(pretrained=False, fusion_mode='concat')
    m_cmfim  = DSCMFNet(pretrained=False, fusion_mode='cmfim')

    p_concat = count_parameters(m_concat)
    p_cmfim  = count_parameters(m_cmfim)

    f_concat = _fusion_param_count(m_concat)
    f_cmfim  = _fusion_param_count(m_cmfim)

    delta_total  = p_cmfim['total']  - p_concat['total']
    delta_fusion = f_cmfim - f_concat

    print(f"  {'':20s}  {'concat':>12s}  {'cmfim':>12s}  {'delta':>12s}")
    print(f"  {'total params':20s}  {p_concat['total']:>12,}  "
          f"{p_cmfim['total']:>12,}  +{delta_total:>11,}")
    print(f"  {'fusion modules':20s}  {f_concat:>12,}  "
          f"{f_cmfim:>12,}  +{delta_fusion:>11,}")
    print(f"  CMFIM overhead: +{delta_total:,} params "
          f"({delta_total / p_concat['total'] * 100:.2f}% of concat total)")

    # ---- Encoder-freeze sanity check ------------------------------------ #
    print(f"\n{'─'*65}")
    print("  freeze_encoder_stages(2) on cmfim model")
    m = DSCMFNet(pretrained=False, fusion_mode='cmfim').train()
    before = count_parameters(m)['trainable']
    m.freeze_encoder_stages(2)
    after  = count_parameters(m)['trainable']
    print(f"    trainable before : {before:,}")
    print(f"    trainable after  : {after:,}")
    print(f"    frozen           : {before - after:,}")
    m.unfreeze_all()
    assert count_parameters(m)['trainable'] == p_cmfim['total']

    print(f"\nSmoke test PASSED.")
    print("=" * 65)


if __name__ == '__main__':
    _smoke_test()
