from typing import Optional, Union, List
from .decoder import UnetDecoder
from ..encoders import get_encoder
from ..base import SegmentationModel
from ..base import SegmentationHead, ClassificationHead


class Unet(SegmentationModel):
    """Unet_ is a fully convolution neural network for image semantic segmentation. Consist of *encoder* 
    and *decoder* parts connected with *skip connections*. Encoder extract features of different spatial 
    resolution (skip connections) which are used by decoder to define accurate segmentation mask. Use *concatenation*
    for fusing decoder blocks with skip connections.

    Args:
        encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution
        encoder_depth: A number of stages used in encoder in range [3, 5]. Each stage generate features 
            two times smaller in spatial dimensions than previous one (e.g. for depth 0 we will have features
            with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
            Default is 5
        encoder_weights: One of **None** (random initialization), **"imagenet"** (pre-training on ImageNet) and 
            other pretrained weights (see table with available weights for each encoder_name)
        decoder_channels: List of integers which specify **in_channels** parameter for convolutions used in decoder.
            Length of the list should be the same as **encoder_depth**
        decoder_use_batchnorm: If **True**, BatchNorm2d layer between Conv2D and Activation layers
            is used. If **"inplace"** InplaceABN will be used, allows to decrease memory consumption.
            Available options are **True, False, "inplace"**
        decoder_attention_type: Attention module used in decoder of the model. Available options are **None** and **scse**.
            SCSE paper - https://arxiv.org/abs/1808.08127
        in_channels: A number of input channels for the model, default is 3 (RGB images)
        classes: A number of classes for output mask (or you can think as a number of channels of output mask)
        activation: An activation function to apply after the final convolution layer.
            Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**, **callable** and **None**.
            Default is **None**
        aux_params: Dictionary with parameters of the auxiliary output (classification head). Auxiliary output is build 
            on top of encoder if **aux_params** is not **None** (default). Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply "sigmoid"/"softmax" (could be **None** to return logits)

    Returns:
        ``torch.nn.Module``: Unet

    .. _Unet:
        https://arxiv.org/abs/1505.04597

    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
        scale_pyramid_module = False, # add
        use_attention_branch=False, # add
        downsample_ratio=1, # add
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels, # "out_channels": (3, 64, 256, 512, 1024, 2048), (resnet50) / (64,128,256,512,512,512) (vgg19_bn)
            decoder_channels=decoder_channels, #  List[int] = (256, 128, 64, 32, 16),
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
            scale_pyramid_module=scale_pyramid_module, # add
            downsample_ratio=downsample_ratio, # add
        )
        # ADD  どこまでupsample・concatするか?デフォルトはinputと同じサイズまで
        i=-1
        if downsample_ratio > 1: # 2,4,8,16
            i = int(downsample_ratio / 2) # 1,2,3,4
            assert i in [1,2,3,4]
            i = -(i+1)
        # TO ADD
        self.segmentation_head = SegmentationHead(
            #in_channels=decoder_channels[-1],
            in_channels=decoder_channels[i],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
            use_attention_branch=use_attention_branch, # add
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.name = "u-{}".format(encoder_name)
        self.initialize()


#class UnetSpm(Unet):
#    """Unet modified to use scale_pyramid_module in decoding process
#    """
#
#    def __init__(
#        self,
#        encoder_name: str = "resnet34",
#        encoder_depth: int = 5,
#        encoder_weights: Optional[str] = "imagenet",
#        decoder_use_batchnorm: bool = True,
#        decoder_channels: List[int] = (256, 128, 64, 32, 16),
#        decoder_attention_type: Optional[str] = None,
#        in_channels: int = 3,
#        classes: int = 1,
#        activation: Optional[Union[str, callable]] = None,
#        aux_params: Optional[dict] = None,
#    ):
# 
#        super(UnetSpm, self).__init__(
#        encoder_name,
#        encoder_depth,
#        encoder_weights,
#        decoder_use_batchnorm,
#        decoder_channels,
#        decoder_attention_type,
#        in_channels,
#        classes,
#        activation,
#        aux_params,
#        )
#
#        self.decoder = UnetDecoderSpm(
#            encoder_channels=self.encoder.out_channels,
#            decoder_channels=decoder_channels,
#            n_blocks=encoder_depth,
#            use_batchnorm=decoder_use_batchnorm,
#            center=True if encoder_name.startswith("vgg") else False,
#            attention_type=decoder_attention_type,
#        )
#