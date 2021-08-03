from typing import Optional, Union, List
from .decoder import UnetDecoder
from ..encoders import get_encoder
from ..base import SegmentationModel
from ..base import SegmentationHead, ClassificationHead
import torch
import torch.nn as nn
import torch.nn.functional as F
#from models.nuclei_detection.models.CCT.decoders import VATDecoder, DropOutDecoder, CutOutDecoder, ContextMaskingDecoder, ObjectMaskingDecoder, FeatureDropDecoder, FeatureNoiseDecoder
from models.CCT.decoders import VATDecoder, DropOutDecoder, CutOutDecoder, ContextMaskingDecoder, ObjectMaskingDecoder, FeatureDropDecoder, FeatureNoiseDecoder



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
        scale_pyramid_module = False,
        use_attention_branch=False,
        downsample_ratio=1,
        deep_supervision=0,
        use_ssl=0,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
            scale_pyramid_module=scale_pyramid_module,
            downsample_ratio=downsample_ratio,
            n_class=classes,
            deep_supervision=deep_supervision,
        )
        i=-1 # default
        if downsample_ratio > 1:
            i = int(downsample_ratio / 2)
            assert i in [1,2,3,4]
            i = -(i+1) 
        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[i],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
            use_attention_branch=use_attention_branch,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.name = "u-{}".format(encoder_name)
        self.deep_supervision = deep_supervision
        if use_ssl:
            tmp = len(decoder_channels)
            if downsample_ratio > 1:
                ii = int(downsample_ratio / 2)
                assert i in [1,2,3,4]
                tmp = tmp - ii
            upscale = 2**tmp
            # The auxilary decoders
            vat_decoder = [VATDecoder(upscale, self.encoder.out_channels[-1], classes, xi=1e-6,
                                        eps=2.0) for _ in range(2)]
            drop_decoder = [DropOutDecoder(upscale, self.encoder.out_channels[-1], classes,
                                        drop_rate=0.5, spatial_dropout=True)
                                        for _ in range(6)]
            #cut_decoder = [CutOutDecoder(upscale, self.encoder.out_channels[-1], classes, erase=conf['erase'])
            #                            for _ in range(conf['cutout'])]
            #context_m_decoder = [ContextMaskingDecoder(upscale, self.encoder.out_channels[-1], classes)
            #                            for _ in range(2)]
            #object_masking = [ObjectMaskingDecoder(upscale, self.encoder.out_channels[-1], classes)
            #                            for _ in range(2)]
            feature_drop = [FeatureDropDecoder(upscale, self.encoder.out_channels[-1], classes)
                                        for _ in range(6)]
            feature_noise = [FeatureNoiseDecoder(upscale, self.encoder.out_channels[-1], classes,
                                        uniform_range=0.3)
                                        for _ in range(6)]

            self.aux_decoders = nn.ModuleList([*vat_decoder, *drop_decoder, *feature_drop, *feature_noise])
            #self.aux_decoders = nn.ModuleList([*vat_decoder, *drop_decoder, *context_m_decoder, *object_masking, *feature_drop, *feature_noise])

 
        else:
            self.aux_decoders = None

        self.initialize()

    # override
    def forward(self, x, unsupervised=False):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)
        if unsupervised:
            # Get main prediction
            if self.deep_supervision:
                output_ul_main, intermediates = self.decoder(*features)
                del intermediates
            else:
                output_ul_main = self.decoder(*features)

            # Get auxiliary predictions
            masks = self.segmentation_head(output_ul_main)

            #outputs_ul_aux = self.aux_decoder(*features)
            masks_aux = [aux_decoder(features[-1], masks.detach()) for aux_decoder in self.aux_decoders]
            del features, x

            return masks, masks_aux

        else: # supervised
            if self.deep_supervision:
                decoder_output, intermediates = self.decoder(*features)
    
                masks = self.segmentation_head(decoder_output)
                del decoder_output
        
                if self.classification_head is not None:
                    labels = self.classification_head(features[-1])
                    return masks, labels, intermediates

                return masks, intermediates
    
            else:
                decoder_output = self.decoder(*features)
    
                masks = self.segmentation_head(decoder_output)
                del decoder_output
    
                if self.classification_head is not None:
                    labels = self.classification_head(features[-1])
                    return masks, labels
        
                return masks

    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        with torch.no_grad():
            if self.deep_supervision:
                masks, intermediates = self.forward(x)
                del intermediates
                return masks

            else:
                if self.classification_head is not None:
                    masks, labels = self.forward(x)
                    return masks, labels

                masks = self.forward(decoder_output)
                return masks





