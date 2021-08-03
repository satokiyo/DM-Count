import torch.nn as nn
#from models.nuclei_detection.models.unet import Unet
from models.unet import Unet


class DMCountModel(nn.Module):
    def __init__(self, args):
        super(DMCountModel, self).__init__()
        encoder_weights=None if args.resume else 'imagenet'
        if encoder_weights is not None:
            if args.encoder_name in ["resnext101_32x4d"]:
                encoder_weights="ssl"
            elif "efficientnetv2" in args.encoder_name:
                encoder_weights=None
        self.model = Unet(
            encoder_name=args.encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=args.classes,
            activation=None, # "sigmoid"/"softmax" (could be **None** to return logits)
            decoder_attention_type=None, # Optional[str] = None or scse
            scale_pyramid_module=args.scale_pyramid_module,
            use_attention_branch=args.use_attention_branch,
            downsample_ratio=args.downsample_ratio,
            deep_supervision=args.deep_supervision,
            use_ssl=args.use_ssl,
        )
        self.use_ssl = args.use_ssl

    def forward(self, input, unsupervised=False):
        x = self.model(input, unsupervised=unsupervised)
        return x

