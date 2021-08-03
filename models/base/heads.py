import torch.nn as nn
from .modules import Flatten, Activation, MyAdaptiveMaxPool2d, MyAdaptiveAvgPool2d
import torch


class SegmentationHead(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1, use_attention_branch=False):
        super(SegmentationHead, self).__init__()
        #self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.conv2d = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        #self.upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        self.upsampling = nn.Identity()
        self.activation = Activation(activation)

        self.use_attention_branch=use_attention_branch
        if self.use_attention_branch:
            self.conv_att = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, 1, kernel_size=1),
                nn.Sigmoid(),
            )

        self.activation2 = nn.Sequential(
             nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
             nn.ReLU(inplace=True),
             nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
             nn.ReLU(inplace=True),
             nn.Conv2d(in_channels, out_channels, kernel_size=1),
             nn.ReLU(inplace=True), # not use activation!
        )



#        else:
#            self.conv_att = torch.one_like()

        #super().__init__(conv2d, upsampling, activation)
    def forward(self, x):
        x = self.conv2d(x)
        x = self.upsampling(x)
        x = self.conv2d(x) # Add
        x = self.conv2d(x) # Add

        if self.use_attention_branch:
            x2 = self.conv_att(x)
            x =  x * x2

        x = self.activation(x)
        x = self.activation2(x)

        return x



class ClassificationHead(nn.Sequential):

    def __init__(self, in_channels, classes, pooling="avg", dropout=0.2, activation=None):
        if pooling not in ("max", "avg"):
            raise ValueError("Pooling should be one of ('max', 'avg'), got {}.".format(pooling))
        #pool = nn.AdaptiveAvgPool2d(1) if pooling == 'avg' else nn.AdaptiveMaxPool2d(1)
        pool = MyAdaptiveAvgPool2d((1,1)) if pooling == 'avg' else MyAdaptiveMaxPool2d((1,1))
        flatten = Flatten()
        dropout = nn.Dropout(p=dropout, inplace=True) if dropout else nn.Identity()
        linear = nn.Linear(in_channels, classes, bias=True)
        activation = Activation(activation)
        super().__init__(pool, flatten, dropout, linear, activation)
