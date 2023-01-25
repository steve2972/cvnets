import torch
from torch import nn, Tensor
from torch.nn import functional as F

from .conv_layer import ConvBlock
from typing import Optional

class SingleHeadAttention(nn.Module):
    """
    This layer applies a single-head attention as described in `DeLighT <https://arxiv.org/abs/2008.00623>`_ paper
    Args:
        embed_dim (int): :math:`C_{in}` from an expected input of size :math:`(N, P, C_{in})`
        attn_dropout (Optional[float]): Attention dropout. Default: 0.0
        bias (Optional[bool]): Use bias or not. Default: ``True``
    Shape:
        - Input: :math:`(N, P, C_{in})` where :math:`N` is batch size, :math:`P` is number of patches,
        and :math:`C_{in}` is input embedding dim
        - Output: same shape as the input
    """

    def __init__(
        self,
        embed_dim: int,
        attn_dropout: Optional[float] = 0.0,
        bias: Optional[bool] = True):
        super().__init__()

        self.qkv_proj = nn.Linear(
            in_features=embed_dim, out_features=3 * embed_dim, bias=bias
        )

        self.attn_dropout = nn.Dropout(p=attn_dropout)
        self.out_proj = nn.Linear(
            in_features=embed_dim, out_features=embed_dim, bias=bias
        )

        self.softmax = nn.Softmax(dim=-1)
        self.embed_dim = embed_dim
        self.scaling = self.embed_dim**-0.5

    def forward(
        self,
        x: Tensor,
        x_kv: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        # [N, P, C] --> [N, P, 3C]
        if x_kv is None:
            qkv = self.qkv_proj(x)
            # [N, P, 3C] --> [N, P, C] x 3
            query, key, value = torch.chunk(qkv, chunks=3, dim=-1)
        else:
            query = F.linear(
                x,
                weight=self.qkv_proj.weight[: self.embed_dim, ...],
                bias=self.qkv_proj.bias[: self.embed_dim],
            )

            # [N, P, C] --> [N, P, 2C]
            kv = F.linear(
                x_kv,
                weight=self.qkv_proj.weight[self.embed_dim :, ...],
                bias=self.qkv_proj.bias[self.embed_dim :],
            )
            key, value = torch.chunk(kv, chunks=2, dim=-1)

        query = query * self.scaling

        # [N, P, C] --> [N, C, P]
        key = key.transpose(-2, -1)

        # QK^T
        # [N, P, C] x [N, C, P] --> [N, P, P]
        attn = torch.matmul(query, key)

        if attn_mask is not None:
            # attn_mask shape should be the same as attn
            assert list(attn_mask.shape) == list(
                attn.shape
            ), "Shape of attention mask and attn should be the same. Got: {} and {}".format(
                attn_mask.shape, attn.shape
            )
            attn = attn + attn_mask

        if key_padding_mask is not None:
            # Do not attend to padding positions
            # key padding mask size is [N, P]
            batch_size, num_src_tokens, num_tgt_tokens = attn.shape
            assert key_padding_mask.dim() == 2 and list(key_padding_mask.shape) == [
                batch_size,
                num_tgt_tokens,
            ], "Key_padding_mask should be 2-dimension with shape [{}, {}]. Got: {}".format(
                batch_size, num_tgt_tokens, key_padding_mask.shape
            )
            attn = attn.masked_fill(
                key_padding_mask.unsqueeze(1).to(torch.bool),
                float("-inf"),
            )

        attn = self.softmax(attn)
        attn = self.attn_dropout(attn)

        # weighted sum
        # [N, P, P] x [N, P, C] --> [N, P, C]
        out = torch.matmul(attn, value)
        out = self.out_proj(out)

        return out


class MultiHeadAttention(nn.Module):
    """
    This layer applies a multi-head self- or cross-attention as described in
    `Attention is all you need <https://arxiv.org/abs/1706.03762>`_ paper
    Args:
        embed_dim (int): :math:`C_{in}` from an expected input of size :math:`(N, S, C_{in})`
        num_heads (int): Number of heads in multi-head attention
        attn_dropout (Optional[float]): Attention dropout. Default: 0.0
        bias (Optional[bool]): Use bias or not. Default: ``True``
    Shape:
        - Input:
           - Query tensor (x_q) :math:`(N, S, C_{in})` where :math:`N` is batch size, :math:`S` is number of source tokens,
        and :math:`C_{in}` is input embedding dim
           - Optional Key-Value tensor (x_kv) :math:`(N, T, C_{in})` where :math:`T` is number of target tokens
        - Output: same shape as the input
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attn_dropout: Optional[float] = 0.0,
        bias: Optional[bool] = True,
        output_dim: Optional[int] = None,
    ) -> None:
        if output_dim is None:
            output_dim = embed_dim
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                "Embedding dim must be divisible by number of heads in {}. Got: embed_dim={} and num_heads={}".format(
                    self.__class__.__name__, embed_dim, num_heads
                )
            )

        self.qkv_proj = nn.Linear(
            in_features=embed_dim, out_features=3 * embed_dim, bias=bias
        )

        self.attn_dropout = nn.Dropout(p=attn_dropout)
        self.out_proj = nn.Linear(
            in_features=embed_dim, out_features=output_dim, bias=bias
        )

        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5
        self.softmax = nn.Softmax(dim=-1)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.use_separate_proj_weight = embed_dim != output_dim

    def forward(
        self,
        x_q: Tensor,
        x_kv: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        out, _ = F.multi_head_attention_forward(
            query=x_q,
            key=x_kv if x_kv is not None else x_q,
            value=x_kv if x_kv is not None else x_q,
            embed_dim_to_check=self.embed_dim,
            num_heads=self.num_heads,
            in_proj_weight=torch.empty([0]),
            in_proj_bias=self.qkv_proj.bias,
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=self.attn_dropout.p,
            out_proj_weight=self.out_proj.weight,
            out_proj_bias=self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            attn_mask=attn_mask,
            use_separate_proj_weight=True,
            q_proj_weight=self.qkv_proj.weight[: self.embed_dim, ...],
            k_proj_weight=self.qkv_proj.weight[
                self.embed_dim : 2 * self.embed_dim, ...
            ],
            v_proj_weight=self.qkv_proj.weight[2 * self.embed_dim :, ...],
        )
        return out

class LinearSelfAttention(nn.Module):
    """
    This layer applies a self-attention with linear complexity, as described in `MobileViTv2 <https://arxiv.org/abs/2206.02680>`_ paper.
    This layer can be used for self- as well as cross-attention.
    Args:
        opts: command line arguments
        embed_dim (int): :math:`C` from an expected input of size :math:`(N, C, H, W)`
        attn_dropout (Optional[float]): Dropout value for context scores. Default: 0.0
        bias (Optional[bool]): Use bias in learnable layers. Default: True
    Shape:
        - Input: :math:`(N, C, P, N)` where :math:`N` is the batch size, :math:`C` is the input channels,
        :math:`P` is the number of pixels in the patch, and :math:`N` is the number of patches
        - Output: same as the input
    .. note::
        For MobileViTv2, we unfold the feature map [B, C, H, W] into [B, C, P, N] where P is the number of pixels
        in a patch and N is the number of patches. Because channel is the first dimension in this unfolded tensor,
        we use point-wise convolution (instead of a linear layer). This avoids a transpose operation (which may be
        expensive on resource-constrained devices) that may be required to convert the unfolded tensor from
        channel-first to channel-last format in case of a linear layer.
    """

    def __init__(
        self,
        embed_dim: int,
        attn_dropout: Optional[float] = 0.0,
    ) -> None:
        super().__init__()

        self.qkv_proj = ConvBlock(
            in_features=embed_dim,
            out_features=1 + (2 * embed_dim),
            kernel_size=1,
            use_norm=False,
            activation="linear"
        )

        self.attn_dropout = nn.Dropout(p=attn_dropout)
        self.out_proj = ConvBlock(
            in_features=embed_dim,
            out_features=embed_dim,
            kernel_size=1,
            use_norm=False,
            activation="linear"
        )
        self.embed_dim = embed_dim

    def _forward_self_attn(self, x: Tensor, *args, **kwargs) -> Tensor:
        # [B, C, P, N] --> [B, h + 2d, P, N]
        qkv = self.qkv_proj(x)

        # Project x into query, key and value
        # Query --> [B, 1, P, N]
        # value, key --> [B, d, P, N]
        query, key, value = torch.split(
            qkv, split_size_or_sections=[1, self.embed_dim, self.embed_dim], dim=1
        )

        # apply softmax along N dimension
        context_scores = F.softmax(query, dim=-1)
        # Uncomment below line to visualize context scores
        # self.visualize_context_scores(context_scores=context_scores)
        context_scores = self.attn_dropout(context_scores)

        # Compute context vector
        # [B, d, P, N] x [B, 1, P, N] -> [B, d, P, N]
        context_vector = key * context_scores
        # [B, d, P, N] --> [B, d, P, 1]
        context_vector = torch.sum(context_vector, dim=-1, keepdim=True)

        # combine context vector with values
        # [B, d, P, N] * [B, d, P, 1] --> [B, d, P, N]
        out = F.relu(value) * context_vector.expand_as(value)
        out = self.out_proj(out)
        return out

    def _forward_cross_attn(
        self, x: Tensor, x_prev: Optional[Tensor] = None, *args, **kwargs
    ) -> Tensor:
        # x --> [B, C, P, N]
        # x_prev = [B, C, P, M]

        batch_size, in_dim, kv_patch_area, kv_num_patches = x.shape

        q_patch_area, q_num_patches = x.shape[-2:]

        assert (
            kv_patch_area == q_patch_area
        ), "The number of pixels in a patch for query and key_value should be the same"

        # compute query, key, and value
        # [B, C, P, M] --> [B, 1 + d, P, M]
        qk = F.conv2d(
            x_prev,
            weight=self.qkv_proj.block.conv.weight[: self.embed_dim + 1, ...],
            bias=self.qkv_proj.block.conv.bias[: self.embed_dim + 1, ...],
        )
        # [B, 1 + d, P, M] --> [B, 1, P, M], [B, d, P, M]
        query, key = torch.split(qk, split_size_or_sections=[1, self.embed_dim], dim=1)
        # [B, C, P, N] --> [B, d, P, N]
        value = F.conv2d(
            x,
            weight=self.qkv_proj.block.conv.weight[self.embed_dim + 1 :, ...],
            bias=self.qkv_proj.block.conv.bias[self.embed_dim + 1 :, ...],
        )

        # apply softmax along M dimension
        context_scores = F.softmax(query, dim=-1)
        context_scores = self.attn_dropout(context_scores)

        # compute context vector
        # [B, d, P, M] * [B, 1, P, M] -> [B, d, P, M]
        context_vector = key * context_scores
        # [B, d, P, M] --> [B, d, P, 1]
        context_vector = torch.sum(context_vector, dim=-1, keepdim=True)

        # combine context vector with values
        # [B, d, P, N] * [B, d, P, 1] --> [B, d, P, N]
        out = F.relu(value) * context_vector.expand_as(value)
        out = self.out_proj(out)
        return out

    def forward(
        self, x: Tensor, x_prev: Optional[Tensor] = None, *args, **kwargs
    ) -> Tensor:
        if x_prev is None:
            return self._forward_self_attn(x, *args, **kwargs)
        else:
            return self._forward_cross_attn(x, x_prev=x_prev, *args, **kwargs)