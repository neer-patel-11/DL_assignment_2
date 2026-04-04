"""Custom IoU loss 
"""

import torch
import torch.nn as nn

class IoULoss(nn.Module):
    """IoU loss for bounding box regression.
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        """
        Initialize the IoULoss module.
        Args:
            eps: Small value to avoid division by zero.
            reduction: Specifies the reduction to apply to the output: 'mean' | 'sum'.
        """
        super().__init__()
        self.eps = eps

        if reduction not in ["none", "mean", "sum"]:
            raise ValueError("reduction must be 'none', 'mean', or 'sum'")
        self.reduction = reduction

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Compute IoU loss between predicted and target bounding boxes.
        Args:
            pred_boxes: [B, 4] predicted boxes in (x_center, y_center, width, height) format.
            target_boxes: [B, 4] target boxes in (x_center, y_center, width, height) format."""
        def to_corners(box):
            x, y, w, h = box[:, 0], box[:, 1], box[:, 2], box[:, 3]
            xmin = x - w / 2
            ymin = y - h / 2
            xmax = x + w / 2
            ymax = y + h / 2
            return xmin, ymin, xmax, ymax

        pxmin, pymin, pxmax, pymax = to_corners(pred_boxes)
        txmin, tymin, txmax, tymax = to_corners(target_boxes)

        # Intersection
        ixmin = torch.max(pxmin, txmin)
        iymin = torch.max(pymin, tymin)
        ixmax = torch.min(pxmax, txmax)
        iymax = torch.min(pymax, tymax)

        inter_w = (ixmax - ixmin).clamp(min=0)
        inter_h = (iymax - iymin).clamp(min=0)
        intersection = inter_w * inter_h

        # Areas
        pred_area = (pxmax - pxmin).clamp(min=0) * (pymax - pymin).clamp(min=0)
        target_area = (txmax - txmin).clamp(min=0) * (tymax - tymin).clamp(min=0)

        union = pred_area + target_area - intersection + self.eps

        iou = intersection / union

        loss = 1 - iou

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
    