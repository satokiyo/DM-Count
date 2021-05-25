# -*- coding: utf-8 -*-
# Author: Guotai Wang
# Date:   12 June, 2020
# Implementation of of COPLENet for COVID-19 pneumonia lesion segmentation from CT images.
# Reference: 
#     G. Wang et al. A Noise-robust Framework for Automatic Segmentation of COVID-19 Pneumonia Lesions 
#     from CT Images. IEEE Transactions on Medical Imaging, 2020. DOI:10.1109/TMI.2020.3000314.

from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import modules as md
from models.hrnet.models.bn_helper import BatchNorm2d, BatchNorm2d_class, relu_inplace
from ..scale_pyramid_module.spm import ScalePyramidModule

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 1):
        super(ConvLayer, self).__init__()
        padding = int((kernel_size - 1) / 2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )
       
    def forward(self, x):
        return self.conv(x)

class SEBlock(nn.Module):
    def __init__(self, in_channels, r):
        super(SEBlock, self).__init__()

        redu_chns = int(in_channels / r)
        self.se_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, redu_chns, kernel_size=1, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(redu_chns, in_channels, kernel_size=1, padding=0),
            nn.ReLU())
        
    def forward(self, x):
        f = self.se_layers(x)
        return f*x + x

class ASPPBlock(nn.Module):
    def __init__(self,in_channels, out_channels_list, kernel_size_list, dilation_list):
        super(ASPPBlock, self).__init__()
        self.conv_num = len(out_channels_list)
        assert(self.conv_num == 4)
        assert(self.conv_num == len(kernel_size_list) and self.conv_num == len(dilation_list))
        pad0 = int((kernel_size_list[0] - 1) / 2 * dilation_list[0])
        pad1 = int((kernel_size_list[1] - 1) / 2 * dilation_list[1])
        pad2 = int((kernel_size_list[2] - 1) / 2 * dilation_list[2])
        pad3 = int((kernel_size_list[3] - 1) / 2 * dilation_list[3])
        self.conv_1 = nn.Conv2d(in_channels, out_channels_list[0], kernel_size = kernel_size_list[0], 
                    dilation = dilation_list[0], padding = pad0 )
        self.conv_2 = nn.Conv2d(in_channels, out_channels_list[1], kernel_size = kernel_size_list[1], 
                    dilation = dilation_list[1], padding = pad1 )
        self.conv_3 = nn.Conv2d(in_channels, out_channels_list[2], kernel_size = kernel_size_list[2], 
                    dilation = dilation_list[2], padding = pad2 )
        self.conv_4 = nn.Conv2d(in_channels, out_channels_list[3], kernel_size = kernel_size_list[3], 
                    dilation = dilation_list[3], padding = pad3 )

        out_channels  = out_channels_list[0] + out_channels_list[1] + out_channels_list[2] + out_channels_list[3] 
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU())
       
    def forward(self, x):
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        x4 = self.conv_4(x)

        y  = torch.cat([x1, x2, x3, x4], dim=1)
        y  = self.conv_1x1(y)
        return y

class ConvBNActBlock(nn.Module):
    """Two convolution layers with batch norm, leaky relu, dropout and SE block"""
    def __init__(self,in_channels, out_channels, dropout_p):
        super(ConvBNActBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            SEBlock(out_channels, 2)
        )
       
    def forward(self, x):
        return self.conv_conv(x)

class DownBlock(nn.Module):
    """Downsampling by a concantenation of max-pool and avg-pool, followed by ConvBNActBlock
    """
    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.avgpool = nn.AvgPool2d(2)
        self.conv    = ConvBNActBlock(2 * in_channels, out_channels, dropout_p)
        
    def forward(self, x):
        x_max = self.maxpool(x)
        x_avg = self.avgpool(x)
        x_cat = torch.cat([x_max, x_avg], dim=1)
        y     = self.conv(x_cat)
        return y + x_cat

class UpBlock(nn.Module):
    """Upssampling followed by ConvBNActBlock"""
    def __init__(self, in_channels1, in_channels2, out_channels, 
                 bilinear=True, dropout_p = 0.5):
        super(UpBlock, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size = 1)
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvBNActBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x1    = self.up(x1)
        x_cat = torch.cat([x2, x1], dim=1)
        y     = self.conv(x_cat)
        return y + x_cat

class COPLENet(nn.Module):
    def __init__(self, 
                 in_channels: int = 3,
                 classes: int = 1,
                 scale_pyramid_module = False, # add
                 use_attention_branch=False, # add
                 downsample_ratio=0, # add
                 deep_supervision=0, # add
                 use_ocr=1, # add
    ):
        super(COPLENet, self).__init__()
        self.in_chns   = in_channels
        self.ft_chns   = (32, 64, 128, 256, 512)
        #self.ft_chns   = (64, 128, 256, 512, 1024)
        self.n_class   = classes
        self.bilinear  = True
        self.dropout   = (0.4, 0.4, 0.4, 0.4, 0.4)
        self.use_attention_branch=use_attention_branch
        self.deep_supervision=deep_supervision
        assert(len(self.ft_chns) == 5)

        f0_half = int(self.ft_chns[0] / 2)
        f1_half = int(self.ft_chns[1] / 2)
        f2_half = int(self.ft_chns[2] / 2)
        f3_half = int(self.ft_chns[3] / 2)
        self.in_conv= ConvBNActBlock(self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1  = DownBlock(self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2  = DownBlock(self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3  = DownBlock(self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4  = DownBlock(self.ft_chns[3], self.ft_chns[4], self.dropout[4])
        
        self.bridge0= ConvLayer(self.ft_chns[0], f0_half)
        self.bridge1= ConvLayer(self.ft_chns[1], f1_half)
        self.bridge2= ConvLayer(self.ft_chns[2], f2_half)
        self.bridge3= ConvLayer(self.ft_chns[3], f3_half)

        self.up1    = UpBlock(self.ft_chns[4], f3_half, self.ft_chns[3], dropout_p = self.dropout[3])
        self.up2    = UpBlock(self.ft_chns[3], f2_half, self.ft_chns[2], dropout_p = self.dropout[2])
        self.up3    = UpBlock(self.ft_chns[2], f1_half, self.ft_chns[1], dropout_p = self.dropout[1])
        self.up4    = UpBlock(self.ft_chns[1], f0_half, self.ft_chns[0], dropout_p = self.dropout[0])

        f4 = self.ft_chns[4]
        aspp_chns = [int(f4 / 4), int(f4 / 4), int(f4 / 4), int(f4 / 4)]
        aspp_knls = [1, 3, 3, 3]
        aspp_dila = [1, 2, 4, 6]
        self.aspp = ASPPBlock(f4, aspp_chns, aspp_knls, aspp_dila)
        
            
        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,  
            kernel_size = 3, padding = 1)


        # ADD  どこまでupsample・concatするか?デフォルトはinputと同じサイズまで
        i=-1 # default
        if downsample_ratio > 1: # 2,4,8,16
            i = int(downsample_ratio / 2) # 1,2,3,4
            assert i in [1,2,3,4]
            i = -(i+1) # -2,-3,-4,-5

        # OCR
        #decoder_channels = (1024, 512, 256, 128, 64)
        decoder_channels = (512, 256, 128, 64, 32)
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

        # spm
        self.center = nn.Identity()

        if scale_pyramid_module:
            self.scale_pyramid_module = ScalePyramidModule(inplanes_aspp=self.ft_chns[-1], inplanes_can=self.ft_chns[-2]) # 1/2^4, 1/2^3
        else:
            self.scale_pyramid_module = nn.Identity()
        
        # deep supervision
        #_out_channels = [head_channels] +  [o for o in out_channels]
        if downsample_ratio > 1:
            decoder_channels = decoder_channels[:-i]
        self.convs=[nn.Conv2d(och, self.n_class, 3, stride=1, padding=1).cuda() for och in decoder_channels[::-1]]
        self.deep_supervision = deep_supervision


        # TO ADD
 
    def forward(self, x):
        x_shape = list(x.shape)
        if(len(x_shape) == 5):
          [N, C, D, H, W] = x_shape
          new_shape = [N*D, C, H, W]
          x = torch.transpose(x, 1, 2)
          x = torch.reshape(x, new_shape)
        x0  = self.in_conv(x)
        x0b = self.bridge0(x0)
        x1  = self.down1(x0)
        x1b = self.bridge1(x1)
        x2  = self.down2(x1)
        x2b = self.bridge2(x2)
        x3  = self.down3(x2)
        x3b = self.bridge3(x3)
        x4  = self.down4(x3)

        # spm
        x0, x1, x2, x3, x4 = self.scale_pyramid_module([x0, x1, x2, x3, x4])
        #x4  = self.aspp(x4) 

        x4 = self.center(x4)

        # decoder
        dx3  = self.up1(x4, x3b)
        dx2   = self.up2(dx3, x2b)
        dx1   = self.up3(dx2, x1b)
        dx0   = self.up4(dx1, x0b)

        # for deep supervision
        out_stack_deep_sup = [] 
        out_stack_deep_sup.append(x4)
        out_stack_deep_sup.append(dx3)
        out_stack_deep_sup.append(dx2)
        out_stack_deep_sup.append(dx1)
        out_stack_deep_sup.append(dx0)

        if self.deep_supervision:
            # allocating deep supervision tensors
            intermediates = []
            # reverse indexing `X_decoder`, so smaller tensors have larger list indices 
            out_stack_deep_sup = out_stack_deep_sup[::-1] # reverse. (512*512*16ch ->...-> 16*16*512ch)
            # deep supervision outputs
            for i in range(1, len(out_stack_deep_sup)): # (256*256*32ch ->...-> 16*16*512ch) !! Final resolution outputはdeep supervisionでは不要 !!
                # 3-by-3 conv2d --> upsampling --> sigmoid output activation
                pool_size = 2**(i)
                hx = self.convs[i](out_stack_deep_sup[i])
                hx = F.interpolate(hx, scale_factor=pool_size, mode="bilinear")
                # collecting deep supervision tensors
                intermediates.append(hx)
            # no need final output

            if self.use_ocr: # OCR
                # ocr
                out_aux = self.aux_head(dx0) # こっちがcoarse object map. class_numのchannelのマップがoutされる
                # compute contrast feature
                feats = self.conv3x3_ocr(dx0) # こっちが普通のinput feature? class_numのchannelに絞る前の、output featureで
        
                context = self.ocr_gather_head(feats, out_aux)
                feats = self.ocr_distri_head(feats, context)
        
                masks = self.out_conv(feats)

                return masks, intermediates, out_aux

            else:
                masks = self.out_conv(dx0)
                return masks, intermediates

        else:

            if self.use_ocr: # OCR
                # ocr
                out_aux = self.aux_head(dx0) # こっちがcoarse object map. class_numのchannelのマップがoutされる
                # compute contrast feature
                feats = self.conv3x3_ocr(dx0) # こっちが普通のinput feature? class_numのchannelに絞る前の、output featureで
        
                context = self.ocr_gather_head(feats, out_aux)
                feats = self.ocr_distri_head(feats, context)

                masks = self.out_conv(feats)

                return masks, out_aux

            else:
                masks = self.out_conv(dx0)
                return masks


#        output = self.out_conv(dx0)
#
#        if(len(x_shape) == 5):
#            new_shape = [N, D] + list(output.shape)[1:]
#            output    = torch.reshape(output, new_shape)
#            output    = torch.transpose(output, 1, 2)
#
#        return output



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



# spm
class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        conv1 = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        super().__init__(conv1, conv2)


