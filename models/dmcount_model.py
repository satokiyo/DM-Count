import torch.nn as nn
from models.unet import Unet
from models.unetplusplus import UnetPlusPlus


class DMCountModel(nn.Module):
    def __init__(self, args):
        super(DMCountModel, self).__init__()

        self.model = Unet( # MAnet UnetPlusPlus DeepLabV3Plus PSPnet
            encoder_name=args.encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=args.classes,
            activation=None, # "sigmoid"/"softmax" (could be **None** to return logits)
            decoder_attention_type=None, # Optional[str] = None or scse
            scale_pyramid_module=args.scale_pyramid_module, # add
            use_attention_branch=args.use_attention_branch, # add
            downsample_ratio=args.downsample_ratio, # add
        )

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

    def forward(self, input):
        x = self.model(input)
        #x = self.activation(x)

        mu = x
        B, C, H, W = mu.size()
        mu_sum = mu.view([B, -1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        mu_normed = mu / (mu_sum + 1e-6)
        return mu, mu_normed


