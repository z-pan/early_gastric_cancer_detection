"""models/losses.py — Shared loss functions for all DSCMFNet model variants.

All three losses are boundary-aware and designed for extremely low-contrast
EGC lesions where standard BCE diverges due to the high background fraction.

Reference loss formulation: CLAUDE.md §Loss Function
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# 1.  Structure Loss (Weighted BCE + Weighted IoU)                            #
# --------------------------------------------------------------------------- #

class StructureLoss(nn.Module):
    """Boundary-weighted BCE + Weighted IoU loss.

    Pixels near the lesion boundary receive up to 6× higher weight, forcing
    the network to learn fine structural detail that is critical for EGC
    where lesion margins are diffuse.

    ``weit = 1 + 5 × |AvgPool(mask) − mask|``

    Automatically upsamples ``pred`` to match ``mask`` if spatial sizes differ
    (e.g. during deep supervision at coarser decoder stages).

    Args:
        pred: ``(B, 1, H', W')`` raw logits.
        mask: ``(B, 1, H,  W)``  binary float mask (0 / 1).

    Returns:
        Scalar loss tensor.
    """

    def forward(self, pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if pred.shape[2:] != mask.shape[2:]:
            pred = F.interpolate(
                pred, size=mask.shape[2:], mode='bilinear', align_corners=True
            )

        # Boundary proximity weight
        weit = 1 + 5 * torch.abs(
            F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask
        )

        # Weighted BCE
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        # Weighted IoU
        pred_sig = torch.sigmoid(pred)
        inter = ((pred_sig * mask) * weit).sum(dim=(2, 3))
        union = ((pred_sig + mask) * weit).sum(dim=(2, 3))
        wiou  = 1 - (inter + 1) / (union - inter + 1)

        return (wbce + wiou).mean()


# --------------------------------------------------------------------------- #
# 2.  Boundary Loss                                                           #
# --------------------------------------------------------------------------- #

class BoundaryLoss(nn.Module):
    """BCE loss on boundary targets derived on-the-fly from the GT mask.

    Boundary GT is obtained via morphological dilation − erosion:

    .. code-block:: python

        dilated    = max_pool2d(mask,  k=3, s=1, p=1)
        eroded     = -max_pool2d(-mask, k=3, s=1, p=1)
        boundary   = (dilated - eroded > 0).float()

    Args:
        bdy_pred: ``(B, 1, H', W')`` raw logits from the boundary head.
        mask:     ``(B, 1, H,  W)``  binary float mask.

    Returns:
        Scalar loss tensor.
    """

    def forward(self, bdy_pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if bdy_pred.shape[2:] != mask.shape[2:]:
            bdy_pred = F.interpolate(
                bdy_pred, size=mask.shape[2:], mode='bilinear', align_corners=True
            )

        dilated  = F.max_pool2d(mask,  kernel_size=3, stride=1, padding=1)
        eroded   = -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)
        bdy_gt   = (dilated - eroded > 0).float()

        return F.binary_cross_entropy_with_logits(bdy_pred, bdy_gt)


# --------------------------------------------------------------------------- #
# 3.  Combined DSCMFNet Loss                                                  #
# --------------------------------------------------------------------------- #

class DSCMFNetLoss(nn.Module):
    """Full training objective combining structure, boundary, and deep supervision.

    .. math::

        L = L_{struct}(seg) + \\lambda_{bdy} \\cdot L_{bdy}(bdy)
          + \\alpha_{ds} \\cdot (0.5 \\cdot L_{struct}(ds_1)
                              + 0.3 \\cdot L_{struct}(ds_2)
                              + 0.2 \\cdot L_{struct}(ds_3))

    where ``ds_1`` is at H/4 (highest weight, most informative),
    ``ds_2`` at H/8, and ``ds_3`` at H/16.

    Args:
        lambda_bdy: Boundary loss weight (default 0.5).
        alpha_ds:   Deep-supervision block weight (default 0.3).
    """

    _DS_WEIGHTS = (0.5, 0.3, 0.2)  # ds_h4, ds_h8, ds_h16

    def __init__(self, lambda_bdy: float = 0.5, alpha_ds: float = 0.3) -> None:
        super().__init__()
        self.structure  = StructureLoss()
        self.boundary   = BoundaryLoss()
        self.lambda_bdy = lambda_bdy
        self.alpha_ds   = alpha_ds

    def forward(
        self,
        seg_pred: torch.Tensor,
        bdy_pred: Optional[torch.Tensor],
        ds_preds: list,
        mask:     torch.Tensor,
    ) -> torch.Tensor:
        """Compute the total training loss.

        Args:
            seg_pred: Main segmentation logits ``(B, 1, H, W)``.
            bdy_pred: Boundary branch logits ``(B, 1, H, W)``, or ``None``
                      when ``fusion_mode='cmfim_no_boundary'`` — boundary
                      loss is skipped in that case.
            ds_preds: 3-element list of auxiliary logits
                      ``[ds_h4, ds_h8, ds_h16]`` at H/4, H/8, H/16.
            mask:     Binary GT mask ``(B, 1, H, W)``.

        Returns:
            Scalar total loss.
        """
        loss = self.structure(seg_pred, mask)
        if bdy_pred is not None:
            loss = loss + self.lambda_bdy * self.boundary(bdy_pred, mask)

        ds_loss = sum(
            w * self.structure(d, mask)
            for w, d in zip(self._DS_WEIGHTS, ds_preds)
        )
        return loss + self.alpha_ds * ds_loss
