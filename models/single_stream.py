"""models/single_stream.py — Single-stream baseline models for ablation study.

Implements three model variants that share the same Progressive Decoder and
output interface as the full DSCMFNet, enabling direct ablation comparison:

  * :class:`SingleStreamResNet34`  — ablation #1: NBI-only, ResNet34 encoder
  * :class:`SingleStreamPVTv2`     — ablation #2: WLI-only, PVTv2-B2 encoder
  * :class:`EarlyFusionPVTv2`      — ablation #3: 6-ch WLI+NBI concat → PVTv2-B2

All three models return ``(seg_pred, bdy_pred, ds_preds)`` matching the
DSCMFNet interface expected by :class:`~models.losses.DSCMFNetLoss`.

Architecture reference: CLAUDE.md §Progressive Decoder, §Segmentation Head,
§Boundary Head, §Deep Supervision.
"""

import warnings
from typing import List, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

from models.losses import DSCMFNetLoss

# --------------------------------------------------------------------------- #
# Shared building blocks                                                      #
# --------------------------------------------------------------------------- #

def _cbg(in_c: int, out_c: int, k: int = 3, p: int = 1) -> nn.Sequential:
    """Conv2d → BatchNorm2d → GELU block.

    Args:
        in_c:  Input channels.
        out_c: Output channels.
        k:     Kernel size (default 3).
        p:     Padding (default 1, keeps spatial size for k=3).

    Returns:
        ``nn.Sequential(Conv2d, BN, GELU)``.
    """
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=k, padding=p, bias=False),
        nn.BatchNorm2d(out_c),
        nn.GELU(),
    )


class _DecoderBlock(nn.Module):
    """Two successive Conv-BN-GELU layers used at each decoder stage.

    Args:
        in_c:  Input channels (after skip-connection concatenation).
        out_c: Output channels.
    """

    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            _cbg(in_c, out_c),
            _cbg(out_c, out_c),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _BoundaryHead(nn.Module):
    """Lightweight boundary prediction head: 32 → 16 → 8 → 1 channels.

    Uses 3×3 convs to preserve spatial structure, followed by a 1×1
    classification conv.

    Args:
        in_c: Input feature channels (default 32, from the final decoder stage).
    """

    def __init__(self, in_c: int = 32) -> None:
        super().__init__()
        self.head = nn.Sequential(
            _cbg(in_c, 16),
            _cbg(16, 8),
            nn.Conv2d(8, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


# --------------------------------------------------------------------------- #
# Progressive Decoder                                                         #
# --------------------------------------------------------------------------- #

class ProgressiveDecoder(nn.Module):
    """Top-down Progressive Decoder with skip connections and deep supervision.

    Receives four feature maps from the encoder (at H/4, H/8, H/16, H/32)
    and progressively upsamples, fusing skip connections at each scale.

    Decoder progression (CLAUDE.md §Progressive Decoder)::

        F4 (C4 × H/32) → 1×1 Conv → 256
          ↑2× + concat(F3) → [Conv-BN-GELU ×2] → 128   ds_h16
          ↑2× + concat(F2) → [Conv-BN-GELU ×2] →  64   ds_h8
          ↑2× + concat(F1) → [Conv-BN-GELU ×2] →  64   ds_h4
          ↑4× → Conv-BN-GELU → 32

    ``F.interpolate(mode='bilinear', align_corners=True)`` is used throughout.
    Size mismatches between upsampled tensor and skip connection are handled
    by re-interpolating to the skip's exact spatial size before concatenation.

    Args:
        enc_channels: 4-tuple ``(C1, C2, C3, C4)`` of encoder output channels
                      at H/4, H/8, H/16, H/32 respectively.
    """

    def __init__(self, enc_channels: Tuple[int, int, int, int]) -> None:
        super().__init__()
        C1, C2, C3, C4 = enc_channels

        # F4 → 256d  (1×1 Conv only, no BN/activation per spec)
        self.proj4 = nn.Conv2d(C4, 256, kernel_size=1)

        # Stage 3: ↑2 + concat(F3) → 128d
        self.stage3 = _DecoderBlock(256 + C3, 128)

        # Stage 2: ↑2 + concat(F2) → 64d
        self.stage2 = _DecoderBlock(128 + C2, 64)

        # Stage 1: ↑2 + concat(F1) → 64d
        self.stage1 = _DecoderBlock(64 + C1, 64)

        # Final ↑4 → 32d
        self.final = _cbg(64, 32)

        # Segmentation and boundary heads (operating on 32d features)
        self.seg_head = nn.Conv2d(32, 1, kernel_size=1)
        self.bdy_head = _BoundaryHead(32)

        # Deep-supervision heads at intermediate scales
        self.ds_head_h16 = nn.Conv2d(128, 1, kernel_size=1)  # H/16
        self.ds_head_h8  = nn.Conv2d(64,  1, kernel_size=1)  # H/8
        self.ds_head_h4  = nn.Conv2d(64,  1, kernel_size=1)  # H/4

    @staticmethod
    def _up(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Upsample ``x`` to match ``target``'s spatial size.

        Args:
            x:      Tensor to upsample.
            target: Tensor whose H×W is the target size.

        Returns:
            Upsampled tensor with the same H×W as ``target``.
        """
        if x.shape[2:] != target.shape[2:]:
            x = F.interpolate(
                x, size=target.shape[2:], mode='bilinear', align_corners=True
            )
        return x

    def forward(
        self,
        f1: torch.Tensor,
        f2: torch.Tensor,
        f3: torch.Tensor,
        f4: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Run the progressive decoder.

        Args:
            f1: Encoder feature at H/4,  channels C1.
            f2: Encoder feature at H/8,  channels C2.
            f3: Encoder feature at H/16, channels C3.
            f4: Encoder feature at H/32, channels C4.

        Returns:
            Tuple of:

            * ``seg_pred``  — ``(B, 1, H, W)`` segmentation logits.
            * ``bdy_pred``  — ``(B, 1, H, W)`` boundary logits.
            * ``ds_preds``  — ``[ds_h4, ds_h8, ds_h16]`` auxiliary logits
              at H/4, H/8, H/16 (same ordering as :attr:`DSCMFNetLoss._DS_WEIGHTS`).
        """
        # ---- F4 → 256d --------------------------------------------------- #
        x = self.proj4(f4)

        # ---- Stage 3: H/16 ----------------------------------------------- #
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self._up(x, f3)
        x = self.stage3(torch.cat([x, f3], dim=1))
        ds_h16 = self.ds_head_h16(x)

        # ---- Stage 2: H/8 ------------------------------------------------ #
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self._up(x, f2)
        x = self.stage2(torch.cat([x, f2], dim=1))
        ds_h8 = self.ds_head_h8(x)

        # ---- Stage 1: H/4 ------------------------------------------------ #
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self._up(x, f1)
        x = self.stage1(torch.cat([x, f1], dim=1))
        ds_h4 = self.ds_head_h4(x)

        # ---- Final ×4 → H ------------------------------------------------ #
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)
        x = self.final(x)

        seg_pred = self.seg_head(x)
        bdy_pred = self.bdy_head(x)

        # ds_preds ordered from shallowest to deepest so that
        # DSCMFNetLoss weights [0.5, 0.3, 0.2] go to [h4, h8, h16]
        return seg_pred, bdy_pred, [ds_h4, ds_h8, ds_h16]


# --------------------------------------------------------------------------- #
# Encoder wrappers                                                            #
# --------------------------------------------------------------------------- #

class _ResNet34Encoder(nn.Module):
    """ResNet34 feature extractor, yielding 4 skip-connection tensors.

    Output channels: ``[64, 128, 256, 512]`` at spatial scales
    ``[H/4, H/8, H/16, H/32]``.

    Args:
        pretrained: Load ImageNet weights when ``True``.
    """

    OUT_CHANNELS: Tuple[int, int, int, int] = (64, 128, 256, 512)

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = tvm.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = tvm.resnet34(weights=weights)

        # Split backbone into reusable sub-modules
        self.stem   = nn.Sequential(backbone.conv1, backbone.bn1,
                                    backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1  # 64ch,  H/4
        self.layer2 = backbone.layer2  # 128ch, H/8
        self.layer3 = backbone.layer3  # 256ch, H/16
        self.layer4 = backbone.layer4  # 512ch, H/32

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract 4-scale features.

        Args:
            x: ``(B, 3, H, W)`` input image.

        Returns:
            ``(c1, c2, c3, c4)`` feature tensors at H/4 … H/32.
        """
        x  = self.stem(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return c1, c2, c3, c4


class _PVTv2B2Encoder(nn.Module):
    """PVTv2-B2 feature extractor via timm, yielding 4 skip-connection tensors.

    Output channels: ``[64, 128, 320, 512]`` at spatial scales
    ``[H/4, H/8, H/16, H/32]``.

    Args:
        pretrained: Load ImageNet weights when ``True``.
        in_chans:   Input image channels.  Pass ``6`` for the early-fusion
                    variant; the first patch-embedding conv is re-initialised
                    to accept 6 channels while inheriting pretrained weights
                    for the 3-channel portion.
    """

    OUT_CHANNELS: Tuple[int, int, int, int] = (64, 128, 320, 512)

    def __init__(self, pretrained: bool = True, in_chans: int = 3) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            'pvt_v2_b2', pretrained=pretrained, features_only=True
        )
        if in_chans != 3:
            self._adapt_input_channels(in_chans)

    def _adapt_input_channels(self, new_in_chans: int) -> None:
        """Replace the first patch-embedding conv to accept ``new_in_chans`` input channels.

        Weight initialisation strategy: the first half of input channels
        copies the pretrained 3-channel weights exactly; the second half
        is initialised to ``pretrained_weight × 0.5`` to keep activation
        magnitudes approximately stable at the start of fine-tuning.

        Args:
            new_in_chans: Number of input channels for the new conv (e.g. 6).
        """
        old_proj = self.backbone.patch_embed.proj   # Conv2d(3, 64, 7, 4, 3)
        old_w = old_proj.weight.data                # (64, 3, 7, 7)
        C_out, _, kH, kW = old_w.shape

        new_proj = nn.Conv2d(
            new_in_chans, C_out,
            kernel_size=old_proj.kernel_size,
            stride=old_proj.stride,
            padding=old_proj.padding,
            bias=old_proj.bias is not None,
        )
        with torch.no_grad():
            new_proj.weight[:, :3, :, :] = old_w           # channels 0-2: full weight
            new_proj.weight[:, 3:, :, :] = old_w * 0.5    # channels 3-5: half weight
            if old_proj.bias is not None:
                new_proj.bias.copy_(old_proj.bias)

        self.backbone.patch_embed.proj = new_proj

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract 4-scale features.

        Args:
            x: ``(B, C, H, W)`` input image (C=3 or C=6).

        Returns:
            ``(c1, c2, c3, c4)`` feature tensors at H/4 … H/32.
        """
        feats = self.backbone(x)
        return feats[0], feats[1], feats[2], feats[3]


# --------------------------------------------------------------------------- #
# Model 1: SingleStreamResNet34  (ablation #1 — NBI only)                    #
# --------------------------------------------------------------------------- #

class SingleStreamResNet34(nn.Module):
    """NBI-only single-stream baseline (ablation experiment #1).

    Encodes a single NBI close-up image with ResNet34 and decodes with the
    same Progressive Decoder used by the full DSCMFNet.

    Args:
        pretrained: Load ImageNet-pretrained encoder weights.

    Input:
        ``x``: ``(B, 3, H, W)`` NBI image.

    Output:
        ``(seg_pred, bdy_pred, ds_preds)``
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        self.encoder = _ResNet34Encoder(pretrained=pretrained)
        self.decoder = ProgressiveDecoder(self.encoder.OUT_CHANNELS)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Forward pass.

        Args:
            x: ``(B, 3, H, W)`` NBI image tensor (ImageNet-normalised).

        Returns:
            ``(seg_pred, bdy_pred, ds_preds)`` where ``seg_pred`` and
            ``bdy_pred`` are ``(B, 1, H, W)`` logits and ``ds_preds`` is
            a list of 3 auxiliary logit tensors.
        """
        c1, c2, c3, c4 = self.encoder(x)
        return self.decoder(c1, c2, c3, c4)


# --------------------------------------------------------------------------- #
# Model 2: SingleStreamPVTv2  (ablation #2 — WLI only)                       #
# --------------------------------------------------------------------------- #

class SingleStreamPVTv2(nn.Module):
    """WLI-only single-stream baseline (ablation experiment #2).

    Encodes a single WLI close-up image with PVTv2-B2 (Pyramid Vision
    Transformer v2, B2 variant) and decodes with the shared Progressive Decoder.

    Args:
        pretrained: Load ImageNet-pretrained encoder weights.

    Input:
        ``x``: ``(B, 3, H, W)`` WLI image.

    Output:
        ``(seg_pred, bdy_pred, ds_preds)``
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        self.encoder = _PVTv2B2Encoder(pretrained=pretrained, in_chans=3)
        self.decoder = ProgressiveDecoder(self.encoder.OUT_CHANNELS)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Forward pass.

        Args:
            x: ``(B, 3, H, W)`` WLI image tensor (ImageNet-normalised).

        Returns:
            ``(seg_pred, bdy_pred, ds_preds)``
        """
        c1, c2, c3, c4 = self.encoder(x)
        return self.decoder(c1, c2, c3, c4)


# --------------------------------------------------------------------------- #
# Model 3: EarlyFusionPVTv2  (ablation #3 — 6-channel concat)               #
# --------------------------------------------------------------------------- #

class EarlyFusionPVTv2(nn.Module):
    """Early-fusion dual-stream baseline (ablation experiment #3).

    Concatenates WLI and NBI images channel-wise into a 6-channel tensor
    and processes it with a modified PVTv2-B2 whose first patch-embedding
    conv has been widened to ``in_chans=6``.

    Pretrained 3-channel weights are re-used for initialisation:
    channels 0-2 copy ImageNet weights exactly; channels 3-5 are
    initialised to half the pretrained weights for stable fine-tuning.

    Args:
        pretrained: Load ImageNet-pretrained encoder weights (3-ch portion).

    Input:
        ``wli``: ``(B, 3, H, W)``  WLI image.
        ``nbi``: ``(B, 3, H, W)``  NBI image.

    Output:
        ``(seg_pred, bdy_pred, ds_preds)``
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        self.encoder = _PVTv2B2Encoder(pretrained=pretrained, in_chans=6)
        self.decoder = ProgressiveDecoder(self.encoder.OUT_CHANNELS)

    def forward(
        self,
        wli: torch.Tensor,
        nbi: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Forward pass.

        Args:
            wli: ``(B, 3, H, W)`` WLI image tensor (ImageNet-normalised).
            nbi: ``(B, 3, H, W)`` NBI image tensor (ImageNet-normalised).

        Returns:
            ``(seg_pred, bdy_pred, ds_preds)``
        """
        x = torch.cat([wli, nbi], dim=1)    # (B, 6, H, W)
        c1, c2, c3, c4 = self.encoder(x)
        return self.decoder(c1, c2, c3, c4)


# --------------------------------------------------------------------------- #
# Dummy forward-pass smoke test                                               #
# --------------------------------------------------------------------------- #

def _smoke_test() -> None:
    """Run a single dummy forward pass for each model variant and print shapes."""
    import sys

    device = torch.device('cpu')
    B, H, W = 2, 352, 352
    x3 = torch.randn(B, 3, H, W, device=device)

    criterion = DSCMFNetLoss()

    # Fake GT mask
    mask = torch.zeros(B, 1, H, W, device=device)
    mask[:, :, 80:180, 100:200] = 1.0

    print(f"\n{'='*65}")
    print("DSCMFNet single-stream smoke test  (pretrained=False)")
    print(f"{'='*65}\n")

    configs = [
        ("SingleStreamResNet34  (ablation #1, NBI-only)",
         SingleStreamResNet34(pretrained=False),
         lambda m: m(x3)),
        ("SingleStreamPVTv2     (ablation #2, WLI-only)",
         SingleStreamPVTv2(pretrained=False),
         lambda m: m(x3)),
        ("EarlyFusionPVTv2      (ablation #3, 6-ch concat)",
         EarlyFusionPVTv2(pretrained=False),
         lambda m: m(x3, x3)),
    ]

    for name, model, fwd in configs:
        model.eval().to(device)
        with torch.no_grad():
            seg, bdy, ds = fwd(model)

        loss = criterion(seg, bdy, ds, mask)

        print(f"  {name}")
        print(f"    seg_pred : {tuple(seg.shape)}  "
              f"range=[{seg.min():.3f}, {seg.max():.3f}]")
        print(f"    bdy_pred : {tuple(bdy.shape)}  "
              f"range=[{bdy.min():.3f}, {bdy.max():.3f}]")
        for i, d in enumerate(ds):
            tag = ['H/4', 'H/8', 'H/16'][i]
            print(f"    ds[{i}] {tag:4s}: {tuple(d.shape)}")
        print(f"    loss     : {loss.item():.4f}")
        print()

    print("Smoke test PASSED.")
    print("=" * 65)


if __name__ == '__main__':
    _smoke_test()
