import torch
from torch import nn, Tensor
from typing import Optional

from layers import ConvBlock, Activation
from utils.math_utils import make_divisible


class SqueezeExcitation(nn.Module):
    """
    This class defines the Squeeze-excitation module, in the `SENet paper <https://arxiv.org/abs/1709.01507>`_
    Args:
        opts: command-line arguments
        in_features (int): :math:`C` from an expected input of size :math:`(N, C, H, W)`
        squeeze_factor (Optional[int]): Reduce :math:`C` by this factor. Default: 4
        scale_fn_name (Optional[str]): Scaling function name. Default: sigmoid
    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)`
    """

    def __init__(
        self,
        in_features: int,
        squeeze_factor: Optional[int] = 4,
        scale_fn_name: Optional[str] = "hswish",
    ) -> None:
        super().__init__()
        squeeze_channels = max(make_divisible(in_features // squeeze_factor, 8), 32)
        act_fn = Activation(scale_fn_name)

        fc1 = ConvBlock(
            in_features,
            squeeze_channels,
            kernel_size=1,
            strides=1,
            use_norm=False,
            activation="relu"
        )

        fc2 = ConvBlock(
            squeeze_channels,
            in_features,
            kernel_size=1,
            strides=1,
            use_norm=False,
            use_act=False
        )

        blocks = [nn.AdaptiveAvgPool2d(output_size=1), fc1, fc2, act_fn]
        self.se_layer = nn.Sequential(*blocks)


        self.in_features = in_features
        self.squeeze_factor = squeeze_factor
        self.scale_fn = scale_fn_name

    def forward(self, x: Tensor) -> Tensor:
        return x * self.se_layer(x)