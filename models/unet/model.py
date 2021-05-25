from typing import Optional, Union, List
from .decoder import UnetDecoder
from ..encoders import get_encoder
from ..base import SegmentationModel
from ..base import SegmentationHead, ClassificationHead
from models.hrnet.models.bn_helper import BatchNorm2d, BatchNorm2d_class, relu_inplace
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        deep_supervision=0, # add
        use_ocr=0, # add
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
            n_class=classes, # add
            deep_supervision=deep_supervision, # add
        )
        # ADD  どこまでupsample・concatするか?デフォルトはinputと同じサイズまで
        i=-1 # default
        if downsample_ratio > 1: # 2,4,8,16
            i = int(downsample_ratio / 2) # 1,2,3,4
            assert i in [1,2,3,4]
            i = -(i+1) # -2,-3,-4,-5

        # OCR
        self.use_ocr=False
        if use_ocr:
            self.use_ocr=True
            last_inp_channels = decoder_channels[i] # UNetだとch数少ない??
#            ocr_mid_channels = 512 #config.MODEL.OCR.MID_CHANNELS
#            ocr_key_channels = 256 #config.MODEL.OCR.KEY_CHANNELS
            ocr_mid_channels = decoder_channels[i] #config.MODEL.OCR.MID_CHANNELS
            ocr_key_channels = int(decoder_channels[i] / 2) #config.MODEL.OCR.KEY_CHANNELS
    
            self.conv3x3_ocr = nn.Sequential(
                nn.Conv2d(last_inp_channels, ocr_mid_channels,
                          kernel_size=3, stride=1, padding=1),
                BatchNorm2d(ocr_mid_channels),
                nn.ReLU(inplace=relu_inplace),
            )
            self.ocr_gather_head = SpatialGather_Module(classes)
    
            self.ocr_distri_head = SpatialOCR_Module(in_channels=ocr_mid_channels,
                                                     key_channels=ocr_key_channels,
                                                     out_channels=ocr_mid_channels,
                                                     scale=1,
                                                     dropout=0.05,
                                                     )
    
            self.aux_head = nn.Sequential(
                nn.Conv2d(last_inp_channels, last_inp_channels,
                          kernel_size=1, stride=1, padding=0),
                BatchNorm2d(last_inp_channels),
                nn.ReLU(inplace=relu_inplace),
                nn.Conv2d(last_inp_channels, classes,
                          kernel_size=1, stride=1, padding=0, bias=True)
            )

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
        self.deep_supervision = deep_supervision
        self.initialize()

    # override
    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)
        if self.deep_supervision:
            decoder_output, intermediate_output = self.decoder(*features)

            if self.use_ocr: # OCR
                # ocr
                out_aux = self.aux_head(decoder_output) # こっちがcoarse object map. class_numのchannelのマップがoutされる
                # compute contrast feature
                feats = self.conv3x3_ocr(decoder_output) # こっちが普通のinput feature? class_numのchannelに絞る前の、output featureで
        
                context = self.ocr_gather_head(feats, out_aux)
                feats = self.ocr_distri_head(feats, context)
        
                #out = self.cls_head(feats)
        
                #out_aux_seg.append(out_aux)
                #out_aux_seg.append(out)
        
                #return out_aux_seg
                masks = self.segmentation_head(feats)

                if self.classification_head is not None:
                    labels = self.classification_head(features[-1])
                    return masks, labels, intermediate_output, out_aux

                return masks, intermediate_output, out_aux

            else:
                masks = self.segmentation_head(decoder_output)
    
            if self.classification_head is not None:
                labels = self.classification_head(features[-1])
                return masks, labels, intermediate_output
    
            return masks, intermediate_output

        else:
            decoder_output = self.decoder(*features)

            if self.use_ocr: # OCR
                # ocr
                out_aux = self.aux_head(decoder_output) # こっちがcoarse object map. class_numのchannelのマップがoutされる
                # compute contrast feature
                feats = self.conv3x3_ocr(decoder_output) # こっちが普通のinput feature? class_numのchannelに絞る前の、output featureで
        
                context = self.ocr_gather_head(feats, out_aux)
                feats = self.ocr_distri_head(feats, context)

                masks = self.segmentation_head(feats)

                if self.classification_head is not None:
                    labels = self.classification_head(features[-1])
                    return masks, labels, out_aux

                return masks, out_aux

            else:
                masks = self.segmentation_head(decoder_output)

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
            #if self.deep_supervision:
            #    x, hx = self.forward(x)
            #    return x, hx
            #else:
            #    x = self.forward(x)
            #    return x

            if self.deep_supervision:
                if self.use_ocr: # OCR
                    if self.classification_head is not None:
                        masks, labels, intermediate_output, out_aux = self.forward(x)
                        return masks, labels, intermediate_output, out_aux
    
                    masks, intermediate_output, out_aux = self.forward(x)
                    return masks, intermediate_output, out_aux
    
                else:
                    masks, intermediate_output = self.forward(x)
                    return masks, intermediate_output
    
            else:
                if self.use_ocr: # OCR
                    if self.classification_head is not None:
                        masks, labels, out_aux = self.forward(x)
                        return masks, labels, out_aux
    
                    masks, out_aux = self.forward(x)
                    return masks, out_aux
    
                else:
                    if self.classification_head is not None:
                        masks, labels = self.forward(x)
                        return masks, labels

                    masks = self.forward(decoder_output)
                    return masks







# for OCR
ALIGN_CORNERS = True
BN_MOMENTUM = 0.1

class ModuleHelper:

    @staticmethod
    def BNReLU(num_features, bn_type=None, **kwargs):
        return nn.Sequential(
            BatchNorm2d(num_features, **kwargs),
            nn.ReLU()
        )

    @staticmethod
    def BatchNorm2d(*args, **kwargs):
        return BatchNorm2d

class SpatialGather_Module(nn.Module):
    """
        Aggregate the context features according to the initial 
        predicted probability distribution.
        Employ the soft-weighted method to aggregate the context.
    """
    def __init__(self, cls_num=0, scale=1):
        super(SpatialGather_Module, self).__init__()
        self.cls_num = cls_num
        self.scale = scale

    # feats : 普通のoutput feature 512ch
    # probs : こっちがcoarse object map. class_numのchannelのマップ
    def forward(self, feats, probs):
        batch_size, c, h, w = probs.size(0), probs.size(1), probs.size(2), probs.size(3)
        probs = probs.view(batch_size, c, -1)
        feats = feats.view(batch_size, feats.size(1), -1)
        feats = feats.permute(0, 2, 1) # batch x hw x c 
        probs = F.softmax(self.scale * probs, dim=2)# batch x k x hw
        ocr_context = torch.matmul(probs, feats)\
        .permute(0, 2, 1).unsqueeze(3)# batch x k x c
        return ocr_context


class _ObjectAttentionBlock(nn.Module):
    '''
    The basic implementation for object context block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
        bn_type           : specify the bn type
    Return:
        N X C X H X W
    '''
    def __init__(self, 
                 in_channels, 
                 key_channels, 
                 scale=1, 
                 bn_type=None):
        super(_ObjectAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_pixel = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_object = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_down = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_up = nn.Sequential(
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.in_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.in_channels, bn_type=bn_type),
        )

    def forward(self, x, proxy):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        query = self.f_pixel(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_object(proxy).view(batch_size, self.key_channels, -1)
        value = self.f_down(proxy).view(batch_size, self.key_channels, -1)
        value = value.permute(0, 2, 1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels**-.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)   

        # add bg context ...
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, *x.size()[2:])
        context = self.f_up(context)
        if self.scale > 1:
            context = F.interpolate(input=context, size=(h, w), mode='bilinear', align_corners=ALIGN_CORNERS)

        return context


class ObjectAttentionBlock2D(_ObjectAttentionBlock):
    def __init__(self, 
                 in_channels, 
                 key_channels, 
                 scale=1, 
                 bn_type=None):
        super(ObjectAttentionBlock2D, self).__init__(in_channels,
                                                     key_channels,
                                                     scale, 
                                                     bn_type=bn_type)


class SpatialOCR_Module(nn.Module):
    """
    Implementation of the OCR module:
    We aggregate the global object representation to update the representation for each pixel.
    """
    def __init__(self, 
                 in_channels, 
                 key_channels, 
                 out_channels, 
                 scale=1, 
                 dropout=0.1, 
                 bn_type=None):
        super(SpatialOCR_Module, self).__init__()
        self.object_context_block = ObjectAttentionBlock2D(in_channels, 
                                                           key_channels, 
                                                           scale, 
                                                           bn_type)
        _in_channels = 2 * in_channels

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(_in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            ModuleHelper.BNReLU(out_channels, bn_type=bn_type),
            nn.Dropout2d(dropout)
        )

    def forward(self, feats, proxy_feats):
        context = self.object_context_block(feats, proxy_feats)

        output = self.conv_bn_dropout(torch.cat([context, feats], 1))

        return output


