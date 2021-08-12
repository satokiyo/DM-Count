import torch.nn as nn
#from models.mobilenet import MobileNetV2

from typing import Optional, Union, List
from .encoders import get_encoder
from .base import ClassificationModelBase
from .base import ClassificationHeadBase

#MODELS = {
#    "mobilenet_v2" : MobileNetV2,
#}


class ClassificationModel(nn.Module):
    def __init__(self, args):
        super(ClassificationModel, self).__init__()
        encoder_weights = None if args.resume else 'imagenet'
        if encoder_weights is not None:
            if args.encoder_name in ["resnext101_32x4d"]:
                encoder_weights="ssl"
            elif "efficientnetv2" in args.encoder_name:
                encoder_weights=None
        self.model = MyModel( 
            encoder_name=args.encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=args.classes,
            activation=args.activation, # Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**, **callable** and **None**.
        )

    def forward(self, input):
        x = self.model(input)
        return x


class MyModel(ClassificationModelBase):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.classification_head = ClassificationHeadBase(
            in_channels=self.encoder.out_channels[-1],
            classes=classes,
            pooling="avg",
            activation=activation,
        )

        if aux_params is not None:
            self.classification_head_aux = ClassificationHeadBase(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head_aux = None

        self.name = "u-{}".format(encoder_name)
        self.initialize()