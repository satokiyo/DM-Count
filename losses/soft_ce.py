from typing import Optional
from torch import nn, Tensor
import torch
import torch.nn.functional as F
from ._functional import label_smoothed_nll_loss

__all__ = ["SoftCrossEntropyLoss"]


class SoftCrossEntropyLoss(nn.Module):

    __constants__ = ["reduction", "ignore_index", "smooth_factor"]

    def __init__(
        self,
        reduction: str = "mean",
        smooth_factor: Optional[float] = None,
        ignore_index: Optional[int] = -100,
        dim: int = 1,
        use_ocr=False,
        w_loss=1.,
        w_loss_ocr=0.4,
    ):
        """Drop-in replacement for torch.nn.CrossEntropyLoss with label_smoothing
        
        Args:
            smooth_factor: Factor to smooth target (e.g. if smooth_factor=0.1 then [1, 0, 0] -> [0.9, 0.05, 0.05])
        
        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W)

        Reference
            https://github.com/BloodAxe/pytorch-toolbelt
        """
        super().__init__()
        self.smooth_factor = smooth_factor
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.dim = dim

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


        log_prob = F.log_softmax(y_pred, dim=self.dim)
        return label_smoothed_nll_loss(
            log_prob,
            y_true,
            epsilon=self.smooth_factor,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            dim=self.dim,
        )

    def forward(self, score, target):
        if not self.use_ocr: # No OCR output.
            score = [score]

        assert len(self.weights) == len(score)

        return sum([w * self._forward(x, target) for (w, x) in zip(self.weights, score)])


