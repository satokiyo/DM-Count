import torch.nn as nn
from models.unet import Unet
from models.unetplusplus import UnetPlusPlus


class DMCountModel(nn.Module):
    def __init__(self, args):
        super(DMCountModel, self).__init__()
        encoder_weights="imagenet"
        if args.encoder_name in ["resnext101_32x4d"]:
            encoder_weights="ssl"
        elif "efficientnetv2" in args.encoder_name:
            encoder_weights=None
        self.model = Unet( # MAnet UnetPlusPlus DeepLabV3Plus PSPnet
            encoder_name=args.encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=args.classes,
            activation=None, # "sigmoid"/"softmax" (could be **None** to return logits)
            decoder_attention_type=None, # Optional[str] = None or scse
            scale_pyramid_module=args.scale_pyramid_module, # add
            use_attention_branch=args.use_attention_branch, # add
            downsample_ratio=args.downsample_ratio, # add
            deep_supervision=args.deep_supervision, # add
            use_ssl=args.use_ssl, # add
        )
        self.use_ssl = args.use_ssl

#        self.model = UnetPlusPlus( # MAnet UnetPlusPlus DeepLabV3Plus PSPnet
#            encoder_name=args.encoder_name,
#            encoder_weights="imagenet",
#            in_channels=3,
#            classes=args.classes,
#            activation=None, # "sigmoid"/"softmax" (could be **None** to return logits)
#            decoder_attention_type=None, # Optional[str] = None or scse
##            scale_pyramid_module=args.scale_pyramid_module, # add
##            use_attention_branch=args.use_attention_branch, # add
#        )



#        self.activation = nn.Sequential(
#            nn.ReLU(inplace=True),
#        )
#        self.activation = nn.Sequential(
#         nn.Conv2d(16, 16, kernel_size=3, padding=1),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(16, 16, kernel_size=3, padding=1),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(16, 1, kernel_size=1),
#         nn.ReLU(inplace=True), # not use activation!
#        )

    def forward(self, input, unsupervised=False):
        x = self.model(input, unsupervised=unsupervised)
        return x

#        mu = x
#        B, C, H, W = mu.size()
#        mu_sum = mu.view([B, -1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
#        mu_normed = mu / (mu_sum + 1e-6)
#        return mu, mu_normed


