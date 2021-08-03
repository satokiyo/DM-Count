import torch
import torch.nn as nn
import functools

if torch.__version__.startswith('0'):
    #from .sync_bn.inplace_abn.bn import InPlaceABNSync
    #BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')
    BatchNorm2d = nn.BatchNorm2d
    #BatchNorm2d_class = InPlaceABNSync
    #BatchNorm2d_class = nn.BatchNorm2d
    relu_inplace = False
else:
    #BatchNorm2d_class = BatchNorm2d = torch.nn.SyncBatchNorm
    BatchNorm2d = nn.BatchNorm2d
    relu_inplace = True