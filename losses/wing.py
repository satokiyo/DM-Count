from typing import Optional
from torch import nn, Tensor
import torch
import torch.nn.functional as F
from ._functional import wing_loss
from .constants import BINARY_MODE, MULTICLASS_MODE, MULTILABEL_MODE

__all__ = ["WingLoss"]


class WingLoss(nn.Module):

    __constants__ = ["reduction", "ignore_index", "smooth_factor"]

    def __init__(
        self,
        mode: str,
        reduction: str = "mean",
        width: Optional[int] = 5,
        curvature: Optional[float] = 0.5,
        ignore_index: Optional[int] = None, 
        use_ocr=False,
        w_loss=1.,
        w_loss_ocr=0.4,
    ):

        assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}

        super().__init__()
        self.mode = mode
        self.reduction = reduction
        self.width = width
        self.curvature = curvature
        self.ignore_index = ignore_index

        self.use_ocr=use_ocr
        if use_ocr:
            self.weights = [w_loss_ocr, w_loss]
        else:
            self.weights = [w_loss]



    def _forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

        ph, pw = y_pred.size(2), y_pred.size(3)
        h, w = y_true.size(1), y_true.size(2)
        if ph != h or pw != w:
            y_pred = F.interpolate(input=y_pred, size=(
                h, w), mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS)

        if self.mode in {BINARY_MODE, MULTILABEL_MODE}:
            pass
            #y_true = y_true.view(-1)
            #y_pred = y_pred.view(-1)

            #if self.ignore_index is not None:
            #    # Filter predictions with ignore label from loss computation
            #    not_ignored = y_true != self.ignore_index
            #    y_pred = y_pred[not_ignored]
            #    y_true = y_true[not_ignored]

            #loss = self.focal_loss_fn(y_pred, y_true)

        elif self.mode == MULTICLASS_MODE:

            num_classes = y_pred.size(1)
            loss = 0

            # Filter anchors with -1 label from loss computation
            if self.ignore_index is not None:
                not_ignored = y_true != self.ignore_index

            y_pred = F.softmax(y_pred, dim=1) # add

            for cls in range(num_classes):
                cls_y_true = (y_true == cls).long()
                cls_y_pred = y_pred[:, cls, ...]

                if self.ignore_index is not None:
                    cls_y_true = cls_y_true[not_ignored]
                    cls_y_pred = cls_y_pred[not_ignored]

                loss += wing_loss(cls_y_pred,
                                       cls_y_true,
                                       reduction=self.reduction,
                                       width=self.width,
                                       curvature=self.curvature,)

        return loss


    def forward(self, score, target):
        if not self.use_ocr: # No OCR output.
            score = [score]

        assert len(self.weights) == len(score)

        return sum([w * self._forward(x, target) for (w, x) in zip(self.weights, score)])


