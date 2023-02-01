import torch
from torch import nn
from typing import Union

from layers import Activation

class ConvBlock(nn.Module):
    def __init__(self,
                in_features:int, 
                out_features:int, 
                kernel_size:int, 
                strides:int=1,
                padding:Union[int, str] = 'same',
                activation:str='relu',
                groups:int=1,
                dilation:int=1,
                use_norm:bool = True,
                use_act:bool = True,
                **kwargs) -> None:
        super().__init__()
        if padding == 'same' and strides == 2:
            padding = kernel_size // 2
        elif padding == 'same' and strides > 2:
            raise NotImplemented("Padding == 'same' not implemented for strides > 2")
        self.use_norm = use_norm
        self.use_act = use_act
        self.conv = nn.Conv2d(
            in_features, out_features, 
            kernel_size, strides, padding, 
            groups=groups, dilation=dilation
        )
        self.bn   = nn.BatchNorm2d(out_features, momentum=0.01)
        self.act = Activation(activation, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        if self.use_norm: x = self.bn(x)
        if self.use_act: x = self.act(x)
        return x
