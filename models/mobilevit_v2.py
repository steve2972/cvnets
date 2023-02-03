from torch import nn

from layers.modules.mobilenet import InvertedResidual
from layers.modules.mobilevit_block import MobileViTBlockv2 as Block
from layers import ConvBlock, GlobalPool, Identity
from .BaseModel import BaseModel

from typing import Tuple

class MobileVitv2(BaseModel):
    def __init__(
        self, 
        num_classes:int=1000):
        super().__init__()

        self.conv_1 = ConvBlock(
            in_features=3,
            out_features=32,
            kernel_size=3,
            strides=2,
            use_norm=True,
        )

        self.layer_1 = make_mobilenet_layer(32, 64, num_blocks=1, expand_ratio=2, strides=1)
        self.layer_2 = make_mobilenet_layer(64, 128,num_blocks=2, expand_ratio=2, strides=2)
        self.layer_3 = make_mit_layer(128, 256, 128, 2, 
            attn_blocks=2, strides=2, expand_ratio=2, 
            ffn_dropout=0.2, attn_dropout=0.2
        )
        self.layer_4 = make_mit_layer(256, 384, 192, 2, 
            attn_blocks=4, strides=2, expand_ratio=2,
            ffn_dropout=0.2, attn_dropout=0.2
        )
        self.layer_5 = make_mit_layer(384, 512, 256, 2, 
            attn_blocks=3, strides=2, expand_ratio=2, 
            ffn_dropout=0.2, attn_dropout=0.2
        )
        self.conv_1x1_exp = Identity()

        self.classifier = nn.Sequential(
            GlobalPool("mean", keep_dim=False),
            nn.Linear(512, num_classes, bias=True)
        )


def make_mobilenet_layer(
        input_channel:int,
        output_channels:int,
        num_blocks:int = 2,
        expand_ratio:int = 4,
        strides:int = 1) -> Tuple[nn.Sequential, int]:
    blocks = []

    for i in range(num_blocks):
        stride = strides if i == 0 else 1
        layer = InvertedResidual(
            in_features=input_channel,
            out_features=output_channels,
            stride=stride,
            expand_ratio=expand_ratio,
        )
        blocks.append(layer)
        input_channel = output_channels

    return nn.Sequential(*blocks)

def make_mit_layer(
    input_channel:int,
    output_channels:int,
    attn_unit_dim:int,
    ffn_multiplier:int,
    patch_h:int = 2,
    patch_w:int = 2,
    dropout:float = 0.0,
    ffn_dropout:float = 0.0,
    attn_dropout:float = 0.0,
    attn_blocks:int = 1,
    expand_ratio:int = 4,
    strides:int = 1,
    prev_dilation:int=1) -> Tuple[nn.Sequential, int]:

    block = []
    stride = strides

    if stride == 2:
        layer = InvertedResidual(
            in_features=input_channel,
            out_features=output_channels,
            stride=stride,
            expand_ratio=expand_ratio,
            dilation=prev_dilation,
        )

        block.append(layer)
        input_channel = output_channels
        

    block.append(
        Block(
            in_features=input_channel,
            attn_unit_dim=attn_unit_dim,
            ffn_multiplier=ffn_multiplier,
            n_attn_blocks=attn_blocks,
            patch_h=patch_h,
            patch_w=patch_w,
            dropout=dropout,
            ffn_dropout=ffn_dropout,
            attn_dropout=attn_dropout,
            conv_ksize=3,
            dilation=prev_dilation,
        )
    )

    return nn.Sequential(*block)