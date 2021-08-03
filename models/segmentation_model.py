import torch.nn as nn
from models.unet import Unet
from itertools import chain


class SegModel(nn.Module):
    def __init__(self, args):
        super(SegModel, self).__init__()
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
            activation=args.activation, # Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**, **callable** and **None**.
            decoder_attention_type=None, # Optional[str] = None or scse
            scale_pyramid_module=args.scale_pyramid_module,
            use_attention_branch=args.use_attention_branch,
            downsample_ratio=args.downsample_ratio,
            deep_supervision=args.deep_supervision,
            use_ocr=args.use_ocr,
            use_ssl=args.use_ssl,
        )
        self.use_ssl = args.use_ssl


    def forward(self, input, unsupervised=False):
        x = self.model(input, unsupervised=unsupervised)
        return x


    def get_backbone_params(self):
        return self.model.encoder.parameters()

    def get_other_params(self):
        if self.use_ssl:
            return chain(self.model.decoder.parameters(), self.model.aux_decoders.parameters())
        return self.model.decoder.parameters()
