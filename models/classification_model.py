import torch.nn as nn
from models.mobilenet import MobileNetV2

MODELS = {
    "mobilenet_v2" : MobileNetV2,
    "timm-mobilenetv3_large_075" : MobileNetV2,
    "timm-mobilenetv3_large_100" : MobileNetV2,
    "timm-mobilenetv3_large_minimal_100" : MobileNetV2,
}


class ClassificationModel(nn.Module):
    def __init__(self, args):
        super(ClassificationModel, self).__init__()
        encoder_weights=None if args.resume else 'imagenet'
        if encoder_weights is not None:
            if args.encoder_name in ["resnext101_32x4d"]:
                encoder_weights="ssl"
            elif "efficientnetv2" in args.encoder_name:
                encoder_weights=None
        self.model = MODELS[args.encoder_name]( 
            encoder_name=args.encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=args.classes,
            activation=args.activation, # Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**, **callable** and **None**.
        )

    def forward(self, input, unsupervised=False):
        x = self.model(input, unsupervised=unsupervised)
        return x