from typing import Optional, Union, List
from ..encoders import get_encoder
from ..base import ClassificationModel
from ..base import ClassificationHead

class MobileNetV2(ClassificationModel):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
        scale_pyramid_module = False,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.classification_head = ClassificationHead(
            in_channels=self.encoder.out_channels[-1],
            classes=classes,
            pooling="avg",
            activation=activation,
        )

        if aux_params is not None:
            self.classification_head_aux = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head_aux = None

        self.name = "u-{}".format(encoder_name)
        self.initialize()

    ## override
    #def forward(self, x, unsupervised=False):
    #    """Sequentially pass `x` trough model`s encoder, decoder and heads"""
    #    features = self.encoder(x)
