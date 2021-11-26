import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import modules as md
from ..scale_pyramid_module.spm import ScalePyramidModule


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
            use_batchnorm=True,
            attention_type=None,
    ):
        super().__init__()
        self.conv1 = md.Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = md.Attention(attention_type, in_channels=in_channels + skip_channels)
        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = md.Attention(attention_type, in_channels=out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="bilinear")
        if skip is not None:
            if x.size()[2:] != skip.size()[2:]:
                skip = F.interpolate(skip, size=x.size()[2:], scale_factor=None, mode="nearest")
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


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


class UnetDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
            n_blocks=5,
            use_batchnorm=True,
            attention_type=None,
            center=False,
            scale_pyramid_module=False,
            downsample_ratio=1,
            n_class=4,
            deep_supervision=1,
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        encoder_channels = encoder_channels[1:]
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0] 
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        if center:
            self.center = CenterBlock(
                head_channels, head_channels, use_batchnorm=use_batchnorm
            )
        else:
            self.center = nn.Identity()

        if scale_pyramid_module:
            self.scale_pyramid_module = ScalePyramidModule(inplanes_aspp=skip_channels[0], inplanes_can=skip_channels[1]) # 1/2^4, 1/2^3

        else:
            self.scale_pyramid_module = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        # どこまでupsample・concatするか?デフォルトはinputと同じサイズまで
        if downsample_ratio > 1:
            i = int(downsample_ratio / 2)
            assert i in [1,2,3,4]
            blocks = blocks[:-i]
        self.blocks = nn.ModuleList(blocks)
        self.n_class=n_class
        _out_channels = [head_channels] +  [o for o in out_channels]
        if downsample_ratio > 1:
            _out_channels = _out_channels[:-i]
        self.deep_supervision = deep_supervision
        if deep_supervision:
            #self.convs=nn.ModuleList([nn.Conv2d(och, self.n_class, 3, stride=1, padding=1) for och in _out_channels[::-1]])
            self.convs=[nn.Conv2d(och, self.n_class, 3, stride=1, padding=1) for och in _out_channels[::-1]]
            self.activations=[nn.ReLU(inplace=True) for _ in _out_channels[::-1]]


    def forward(self, *features):

        features = self.scale_pyramid_module(features) # spm

        features = features[1:]    # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]
        del features


        x = self.center(head)
        if self.deep_supervision:
            out_stack_deep_sup = [] # for deep supervision
            out_stack_deep_sup.append(x)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)
            if self.deep_supervision:
                out_stack_deep_sup.append(x)
            del skip

        if self.deep_supervision:
            # allocating deep supervision tensors
            intermediates = []
            # reverse indexing `X_decoder`, so smaller tensors have larger list indices 
            out_stack_deep_sup = out_stack_deep_sup[::-1] # reverse.(512*512*16ch ->...-> 16*16*512ch)
            # deep supervision outputs
            i=1
            for out_stack in out_stack_deep_sup[1:]:
                # 3-by-3 conv2d --> upsampling --> sigmoid output activation
                pool_size = 2**(i)
                hx = self.convs[i].to(x.device)(out_stack)
                hx = self.activations[i].to(x.device)(hx)
                del out_stack
                hx = F.interpolate(hx, scale_factor=pool_size, mode="bilinear")
                # collecting deep supervision tensors
                intermediates.append(hx)
                del hx
                i+=1
            # no need final output
            del out_stack_deep_sup

            return x, intermediates

        else:
            return x
