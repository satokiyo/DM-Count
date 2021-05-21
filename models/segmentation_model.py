import torch.nn as nn
from models.unet import Unet
from models.unetplusplus import UnetPlusPlus
from models.deeplabv3 import DeepLabV3Plus


class SegModel(nn.Module):
    def __init__(self, args):
        super(SegModel, self).__init__()

        self.model = Unet( # MAnet UnetPlusPlus DeepLabV3Plus PSPnet
            encoder_name=args.encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=args.classes,
            activation=args.activation, # Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**, **callable** and **None**.
            decoder_attention_type=None, # Optional[str] = None or scse
            scale_pyramid_module=args.scale_pyramid_module, # add
            use_attention_branch=args.use_attention_branch, # add
            downsample_ratio=args.downsample_ratio, # add
            deep_supervision=args.deep_supervision, # add
        )

#        self.model = UnetPlusPlus( # MAnet UnetPlusPlus DeepLabV3Plus PSPnet
#            encoder_name=args.encoder_name,
#            encoder_weights="imagenet",
#            in_channels=3,
#            classes=args.classes,
#            activation=args.activation, # "sigmoid"/"softmax" (could be **None** to return logits)
##            decoder_attention_type=None, # Optional[str] = None or scse
##            scale_pyramid_module=args.scale_pyramid_module, # add
##            use_attention_branch=args.use_attention_branch, # add
#        )


    def forward(self, input):
        x = self.model(input)
        return x


