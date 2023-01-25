from torch import nn, Tensor
from typing import Optional

from layers import (
    ConvBlock,
    MultiHeadAttention,
    SingleHeadAttention,
    LinearSelfAttention
)

class TransformerEncoder(nn.Module):
    """
    This class defines the pre-norm `Transformer encoder <https://arxiv.org/abs/1706.03762>`_
    Args:
        opts: command line arguments
        embed_dim (int): :math:`C_{in}` from an expected input of size :math:`(N, P, C_{in})`
        ffn_latent_dim (int): Inner dimension of the FFN
        num_heads (Optional[int]) : Number of heads in multi-head attention. Default: 8
        attn_dropout (Optional[float]): Dropout rate for attention in multi-head attention. Default: 0.0
        dropout (Optional[float]): Dropout rate. Default: 0.0
        ffn_dropout (Optional[float]): Dropout between FFN layers. Default: 0.0
        transformer_norm_layer (Optional[str]): Normalization layer. Default: layer_norm

    Shape:
        - Input: :math:`(N, P, C_{in})` where :math:`N` is batch size, :math:`P` is number of patches,
        and :math:`C_{in}` is input embedding dim
        - Output: same shape as the input
    """

    def __init__(
        self,
        embed_dim: int,
        ffn_latent_dim: int,
        num_heads: Optional[int] = 8,
        attn_dropout: Optional[float] = 0.0,
        dropout: Optional[float] = 0.0,
        ffn_dropout: Optional[float] = 0.0,
        transformer_norm_layer: Optional[str] = "layer_norm",
    ) -> None:

        super().__init__()

        attn_unit = SingleHeadAttention(
            embed_dim=embed_dim, attn_dropout=attn_dropout
        )
        if num_heads > 1:
            attn_unit = MultiHeadAttention(
                embed_dim,
                num_heads,
                attn_dropout=attn_dropout,
            )

        self.pre_norm_mha = nn.Sequential(
            nn.BatchNorm1d(embed_dim),
            attn_unit,
            nn.Dropout(p=dropout),
        )

        act_name = nn.ReLU()
        self.pre_norm_ffn = nn.Sequential(
            nn.BatchNorm1d(embed_dim),
            nn.Linear(in_features=embed_dim, out_features=ffn_latent_dim, bias=True),
            act_name,
            nn.Dropout(p=ffn_dropout),
            nn.Linear(in_features=ffn_latent_dim, out_features=embed_dim, bias=True),
            nn.Dropout(p=dropout),
        )
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_latent_dim
        self.ffn_dropout = ffn_dropout
        self.std_dropout = dropout
        self.attn_fn_name = attn_unit.__class__.__name__
        self.act_fn_name = act_name.__class__.__name__
        self.norm_type = transformer_norm_layer

    def forward(
        self,
        x: Tensor,
        x_prev: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:

        # Multi-head attention
        res = x
        x = self.pre_norm_mha[0](x)  # norm
        x = self.pre_norm_mha[1](
            x_q=x,
            x_kv=x_prev,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask
        )  # mha
        x = self.pre_norm_mha[2](x)  # dropout
        x = x + res

        # Feed forward network
        x = x + self.pre_norm_ffn(x)
        return x


class LinearAttnFFN(nn.Module):
    """
    This class defines the pre-norm transformer encoder with linear self-attention in `MobileViTv2 <https://arxiv.org/abs/2206.02680>`_ paper
    Args:
        opts: command line arguments
        embed_dim (int): :math:`C_{in}` from an expected input of size :math:`(B, C_{in}, P, N)`
        ffn_latent_dim (int): Inner dimension of the FFN
        attn_dropout (Optional[float]): Dropout rate for attention in multi-head attention. Default: 0.0
        dropout (Optional[float]): Dropout rate. Default: 0.0
        ffn_dropout (Optional[float]): Dropout between FFN layers. Default: 0.0
        norm_layer (Optional[str]): Normalization layer. Default: layer_norm_2d

    Shape:
        - Input: :math:`(B, C_{in}, P, N)` where :math:`B` is batch size, :math:`C_{in}` is input embedding dim,
            :math:`P` is number of pixels in a patch, and :math:`N` is number of patches,
        - Output: same shape as the input
    """

    def __init__(
        self,
        embed_dim: int,
        ffn_latent_dim: int,
        attn_dropout: Optional[float] = 0.0,
        dropout: Optional[float] = 0.1,
        ffn_dropout: Optional[float] = 0.0,
    ) -> None:
        super().__init__()
        attn_unit = LinearSelfAttention(embed_dim, attn_dropout)

        self.pre_norm_attn = nn.Sequential(
            nn.BatchNorm2d(embed_dim),
            attn_unit,
            nn.Dropout(p=dropout),
        )

        self.pre_norm_ffn = nn.Sequential(
            nn.BatchNorm2d(embed_dim),
            ConvBlock(
                in_features=embed_dim,
                out_features=ffn_latent_dim,
                kernel_size=1,
                stride=1,
                use_norm=False,
            ),
            nn.Dropout(p=ffn_dropout),
            ConvBlock(
                in_features=ffn_latent_dim,
                out_features=embed_dim,
                kernel_size=1,
                stride=1,
                use_norm=False,
                activation="linear",
            ),
            nn.Dropout(p=dropout),
        )

        self.embed_dim = embed_dim
        self.ffn_dim = ffn_latent_dim
        self.ffn_dropout = ffn_dropout
        self.std_dropout = dropout
        self.attn_fn_name = attn_unit.__repr__()


    def forward(
        self, x: Tensor, x_prev: Optional[Tensor] = None) -> Tensor:
        if x_prev is None:
            # self-attention
            x = x + self.pre_norm_attn(x)
        else:
            # cross-attention
            res = x
            x = self.pre_norm_attn[0](x)  # norm
            x = self.pre_norm_attn[1](x, x_prev)  # attn
            x = self.pre_norm_attn[2](x)  # drop
            x = x + res  # residual

        # Feed forward network
        x = x + self.pre_norm_ffn(x)
        return x