import torch
import torch.nn as nn
import torch.nn.functional as F
from .constants import BINARY_MODE, MULTICLASS_MODE, MULTILABEL_MODE

ALPHA = 0.5 # < 0.5 penalises FP more, > 0.5 penalises FN more
CE_RATIO = 0.5 #weighted contribution of modified CE loss compared to Dice loss

def combo_loss(inputs, targets, smooth=1.0, alpha=0.5, ce_ratio=0.5, from_logits=True, mode='multiclass', ignore_index=None, log_loss=False, classes=None):
    """
    alpha : control FN/FP. 0.5 以下はFP penalize, 0.5以上はFN penalize
    ce_ratio : control contribution ratio of Dice / BCE
    """

#    height, width = targets.shape[1:]
#    n_batch=inputs.size()[0]
#    n_class=inputs.size()[1]
#    segmentation_mask = torch.zeros((n_batch, n_class, height, width), dtype=torch.float32).cuda() # channel first
##    for i_batch in range(n_batch):
##        for i_cls in range(n_class):
##            segmentation_mask[i_batch, i_cls, :, :] = (targets[i_batch] == i_cls).float() 
##    targets=segmentation_mask
#
#    for cls in range(4):
#        cls_y_true = (targets == cls).long()
#        segmentation_mask[:, cls, ...] = cls_y_true
#    targets = segmentation_mask
# 
#
#    #flatten label and prediction tensors
#    inputs = inputs.view(-1)
#    targets = targets.view(-1)
    
#    #True Positives, False Positives & False Negatives
#    intersection = (inputs * targets).sum()    
#    dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
#    
#    e = 1e-7
#    inputs = torch.clamp(inputs, e, 1.0 - e)       
#    #out = - (alpha * ((targets * torch.log(inputs)) + ((1 - alpha) * (1.0 - targets) * torch.log(1.0 - inputs)))) # bug?
#    #out = - (alpha * ((targets - torch.log(inputs)) + ((1 - alpha) * (1.0 - targets) * torch.log(1.0 - inputs))))
#    out = (alpha * (targets - torch.log(inputs))) + ((1 - alpha) * (1.0 - targets) * torch.log(1.0 - inputs))
#    weighted_ce = - out.mean(-1)
#    combo = (ce_ratio * weighted_ce) - ((1 - ce_ratio) * dice)
#
#    return combo

    y_pred = inputs
    y_true = targets
    ph, pw = y_pred.size(2), y_pred.size(3)
    h, w = y_true.size(1), y_true.size(2)
    if ph != h or pw != w:
        y_pred = F.interpolate(input=y_pred, size=(
            h, w), mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS)
    if from_logits:
        # Apply activations to get [0..1] class probabilities
        # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
        # extreme values 0 and 1
        if mode == MULTICLASS_MODE:
            y_pred = y_pred.log_softmax(dim=1).exp()
        else:
            y_pred = F.logsigmoid(y_pred).exp()
    bs = y_true.size(0)
    num_classes = y_pred.size(1)
    dims = (0, 2)
    if mode == BINARY_MODE:
        y_true = y_true.view(bs, 1, -1)
        y_pred = y_pred.view(bs, 1, -1)
        if ignore_index is not None:
            mask = y_true != self.ignore_index
            y_pred = y_pred * mask
            y_true = y_true * mask
    if mode == MULTICLASS_MODE:
        y_true = y_true.view(bs, -1)
        y_pred = y_pred.view(bs, num_classes, -1)
        if ignore_index is not None:
            mask = y_true != ignore_index
            y_pred = y_pred * mask.unsqueeze(1)
            y_true = F.one_hot((y_true * mask).to(torch.long), num_classes)  # N,H*W -> N,H*W, C
            y_true = y_true.permute(0, 2, 1) * mask.unsqueeze(1)  # H, C, H*W
        else:
            y_true = F.one_hot(y_true, num_classes)  # N,H*W -> N,H*W, C
            y_true = y_true.permute(0, 2, 1)  # H, C, H*W
    if mode == MULTILABEL_MODE:
        y_true = y_true.view(bs, num_classes, -1)
        y_pred = y_pred.view(bs, num_classes, -1)
        if ignore_index is not None:
            mask = y_true != ignore_index
            y_pred = y_pred * mask
            y_true = y_true * mask

    #gamma=2
    #if dims is not None:
    #    tmp0 = torch.sum(torch.pow(torch.abs(y_true - y_pred), gamma), dim=dims)
    #    tmp1 = torch.sum(y_pred, dim=dims)
    #    tmp2 = torch.sum(y_true, dim=dims)
    #dice = tmp0 / (tmp1 + tmp2) # nr_dice_loss
    #gamma=1
    #if dims is not None:
    #    tmp0 = torch.sum(torch.pow(torch.abs(y_true - y_pred), gamma), dim=dims)
    #    tmp1 = torch.sum(y_pred, dim=dims)
    #    tmp2 = torch.sum(y_true, dim=dims)
    #w_mae = tmp0 / (tmp1 + tmp2) # nr_dice_loss
    e = 1e-7
    ce_w = alpha
    ce_d_w = ce_ratio
    y_true_f = y_true
    y_pred_f = y_pred

    weighted_ce = ((ce_w * (y_true_f - torch.log(y_pred_f))) + ((1 - ce_w) * (1.0 - y_true_f) * torch.log(1.0 - y_pred_f))).mean()
    #weighted_ce = torch.sum((ce_w * (y_true_f - torch.log(y_pred_f))) + ((1 - ce_w) * (1.0 - y_true_f) * torch.log(1.0 - y_pred_f)), dim=(0,2)) / y_true_f.size(2)
    #weighted_ce = weighted_ce.mean()

    eps=1e-7
    intersection = torch.sum(y_pred_f * y_true_f, dim=dims)
    cardinality = torch.sum(y_pred_f + y_true_f, dim=dims)
    dice_score = (2.0 * intersection + smooth) / (cardinality + smooth).clamp_min(eps)

    if log_loss:
        dice = -torch.log(dice_score.clamp_min(eps))
    else:
        dice = 1.0 - dice_score
    # Dice dice is undefined for non-empty classes
    # So we zero contribution of channel that does not have true pixels
    # NOTE: A better workaround would be to use dice term `mean(y_pred)`
    # for this case, however it will be a modified jaccard dice
    mask = y_true_f.sum(dims) > 0
    dice *= mask.to(dice.dtype)
    if classes is not None:
        dice = dice[classes]
    dice = dice.mean()

    #combo = (ce_d_w * weighted_ce) - ((1 - ce_d_w) * dice)
    combo = (ce_d_w * weighted_ce) + ((1 - ce_d_w) * dice)

    return combo













class ComboLoss(nn.Module):
    def __init__(self,
                 loss_weight=1.0,
                 smooth=1,
                 alpha=0.5,
                 ce_ratio=0.5,
                 reduction='mean',
                 class_weight=None, 
                 use_sigmoid=True,
                 ):
        super(ComboLoss, self).__init__()
        self.loss_weight=loss_weight
        self.smooth=smooth
        self.alpha=alpha
        self.ce_ratio=ce_ratio
        self.reduction=reduction
        self.use_sigmoid=use_sigmoid


    def forward(self, inputs, targets):

        # apply sigmoid
        #inputs = torch.sigmoid(inputs, dim=1)
#        inputs = F.softmax(inputs, dim=1)
        closs = combo_loss(
#            inputs, targets, smooth=self.smooth, alpha=self.alpha, ce_ratio=self.ce_ratio) # tmp for check
            inputs, targets, smooth=self.smooth, alpha=self.alpha, ce_ratio=1.0) # equals to cross entropy loss
        loss = self.loss_weight * closs

        return loss
