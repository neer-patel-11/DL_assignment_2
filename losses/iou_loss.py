import torch
import torch.nn as nn


class IoULoss(nn.Module):
    """
    Generalized IoU Loss (GIoU)
    Works better than vanilla IoU
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        super().__init__()
        self.eps = eps

        if reduction not in ["none", "mean", "sum"]:
            raise ValueError("reduction must be 'none', 'mean', or 'sum'")
        self.reduction = reduction

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor):

        def to_corners(box):
            x, y, w, h = box[:, 0], box[:, 1], box[:, 2], box[:, 3]
            xmin = x - w / 2
            ymin = y - h / 2
            xmax = x + w / 2
            ymax = y + h / 2
            return xmin, ymin, xmax, ymax

        pxmin, pymin, pxmax, pymax = to_corners(pred_boxes)
        txmin, tymin, txmax, tymax = to_corners(target_boxes)

        # ---------- INTERSECTION ----------
        ixmin = torch.max(pxmin, txmin)
        iymin = torch.max(pymin, tymin)
        ixmax = torch.min(pxmax, txmax)
        iymax = torch.min(pymax, tymax)

        inter_w = (ixmax - ixmin).clamp(min=0)
        inter_h = (iymax - iymin).clamp(min=0)
        intersection = inter_w * inter_h

        # ---------- AREAS ----------
        pred_area = (pxmax - pxmin).clamp(min=0) * (pymax - pymin).clamp(min=0)
        target_area = (txmax - txmin).clamp(min=0) * (tymax - tymin).clamp(min=0)

        union = pred_area + target_area - intersection + self.eps
        iou = intersection / union

        # ---------- ENCLOSING BOX ----------
        cxmin = torch.min(pxmin, txmin)
        cymin = torch.min(pymin, tymin)
        cxmax = torch.max(pxmax, txmax)
        cymax = torch.max(pymax, tymax)

        c_area = (cxmax - cxmin).clamp(min=0) * (cymax - cymin).clamp(min=0) + self.eps

        # ---------- GIoU ----------
        giou = iou - (c_area - union) / c_area

        loss = 1 - giou

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss