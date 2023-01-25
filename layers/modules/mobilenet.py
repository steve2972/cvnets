from torch import nn, Tensor
from typing import Optional, Union

from layers import ConvBlock
from .squeeze_excitation import SqueezeExcitation
from utils.math_utils import make_divisible


class InvertedResidualSE(nn.Module):
    """
    This class implements the inverted residual block with squeeze-excitation unit, as described in
    `MobileNetv3 <https://arxiv.org/abs/1905.02244>`_ paper
    Args:
        opts: command-line arguments
        in_features (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H_{in}, W_{in})`
        out_features (int): :math:`C_{out}` from an expected output of size :math:`(N, C_{out}, H_{out}, W_{out)`
        expand_ratio (Union[int, float]): Expand the input channels by this factor in depth-wise conv
        dilation (Optional[int]): Use conv with dilation. Default: 1
        stride (Optional[int]): Use convolutions with a stride. Default: 1
        use_se (Optional[bool]): Use squeeze-excitation block. Default: False
        activation (Optional[str]): Activation function name. Default: relu
        se_scale_fn_name (Optional [str]): Scale activation function inside SE unit. Defaults to hard_sigmoid
        kernel_size (Optional[int]): Kernel size in depth-wise convolution. Defaults to 3.
        squeeze_factor (Optional[bool]): Squeezing factor in SE unit. Defaults to 4.
    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})`
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        expand_ratio: Union[int, float],
        dilation: Optional[int] = 1,
        stride: Optional[int] = 1,
        use_se: Optional[bool] = False,
        activation: Optional[str] = "relu",
        se_scale_fn_name: Optional[str] = "hsigmoid",
        kernel_size: Optional[int] = 3,
        squeeze_factor: Optional[int] = 4):
        super().__init__()
        
        hidden_dim = make_divisible(int(round(in_features * expand_ratio)), 8)

        blocks = []
        if expand_ratio != 1:
            blocks.append(ConvBlock(
                in_features,
                hidden_dim,
                kernel_size=1,
                activation=activation
            ))
        
        blocks.append(ConvBlock(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            strides=stride,
            activation=activation,
            groups=hidden_dim,
            dilation=dilation,
        ))


        if use_se:
            se = SqueezeExcitation(
                in_features=hidden_dim,
                squeeze_factor=squeeze_factor,
                scale_fn_name=se_scale_fn_name,
            )
            blocks.append(se)

        blocks.append(ConvBlock(
            hidden_dim,
            out_features,
            kernel_size=1,
            use_act=False
        ))

        self.block = nn.Sequential(*blocks)
        self.in_features = in_features
        self.out_features = out_features
        self.exp = expand_ratio
        self.dilation = dilation
        self.use_se = use_se
        self.stride = stride
        self.activation = activation
        self.kernel_size = kernel_size
        self.use_res_connect = self.stride == 1 and in_features == out_features

    def forward(self, x: Tensor) -> Tensor:
        y = self.block(x)
        return x + y if self.use_res_connect else y

class InvertedResidual(nn.Module):
    """
    This class implements the inverted residual block, as described in `MobileNetv2 <https://arxiv.org/abs/1801.04381>`_ paper
    Args:
        opts: command-line arguments
        in_features (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H_{in}, W_{in})`
        out_features (int): :math:`C_{out}` from an expected output of size :math:`(N, C_{out}, H_{out}, W_{out)`
        stride (Optional[int]): Use convolutions with a stride. Default: 1
        expand_ratio (Union[int, float]): Expand the input channels by this factor in depth-wise conv
        dilation (Optional[int]): Use conv with dilation. Default: 1
        skip_connection (Optional[bool]): Use skip-connection. Default: True
    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})`
    .. note::
        If `in_features =! out_features` and `stride > 1`, we set `skip_connection=False`
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        stride: int,
        expand_ratio: Union[int, float],
        dilation: int = 1,
        skip_connection: Optional[bool] = True,
    ) -> None:
        assert stride in [1, 2]
        hidden_dim = make_divisible(int(round(in_features * expand_ratio)), 8)

        super().__init__()

        blocks = []

        if expand_ratio != 1:
            blocks.append(ConvBlock(
                    in_features=in_features,
                    out_features=hidden_dim,
                    kernel_size=1,
                    use_norm=True,
                ),
            )

        blocks.append(ConvBlock(
            in_features=hidden_dim,
            out_features=hidden_dim,
            strides=stride,
            kernel_size=3,
            groups=hidden_dim,
            use_norm=True,
            dilation=dilation,
        ))

        blocks.append(ConvBlock(
            in_features=hidden_dim,
            out_features=out_features,
            kernel_size=1,
            activation="linear",
            use_norm=True,
        ))


        self.block = nn.Sequential(*blocks)
        self.in_features = in_features
        self.out_features = out_features
        self.exp = expand_ratio
        self.dilation = dilation
        self.stride = stride
        self.use_res_connect = (
            self.stride == 1 and in_features == out_features and skip_connection
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)