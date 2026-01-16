"""Standard layers."""

from __future__ import annotations

from typing import Optional, Tuple

from jaxtyping import Float, Int
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import MaskedHSTUAttention, MaskedSelfAttentionWithRoPE
from .feedforward import SwiGLU
from .layernorm import RMSNorm

__all__ = [
    "LlamaDecoderLayer",
    "SequentialTransductionUnit",
]


class LlamaDecoderLayer(nn.Module):
    """A standard Llama Transformer Decoder Layer, following `transformers.LlamaDecoderLayer`'s
    implementation."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        attention_dropout: float = 0.0,
        attention_bias: bool = False,
        ffn_bias: bool = False,
    ) -> None:
        """Initializes StandardDecoderLayer module.

        Args:
            hidden_size (int): Dimensionality of the model's hidden representations.
            num_heads (int): Number of attention heads.
            intermediate_size (int): Dimensionality of the feed-forward network's intermediate layer.
            attention_dropout (float): Dropout rate for attention weights. Default is 0.0.
            attention_bias (bool): Whether to include bias terms in the attention projections. Default is False.
            ffn_bias (bool): Whether to include bias terms in the feed-forward network projections. Default is False.
        """
        super().__init__()

        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads."
        self.head_dim = hidden_size // num_heads
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.attention_dropout = attention_dropout
        self.attention_bias = attention_bias
        self.ffn_bias = ffn_bias

        self.self_attn = MaskedSelfAttentionWithRoPE(
            hidden_size=hidden_size,
            head_dim=self.head_dim,
            num_heads=num_heads,
            attention_dropout=attention_dropout,
            attention_bias=attention_bias,
        )
        self.mlp = SwiGLU(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            ffn_bias=ffn_bias,
        )
        self.input_layernorm = RMSNorm(hidden_size)
        self.post_attention_layernorm = RMSNorm(hidden_size)

    def forward(
        self,
        hidden_states: Float[torch.Tensor, "B L d"],
        attention_mask: Optional[Float[torch.Tensor, "B 1 L L"]] = None,
        position_embeddings: Optional[
            Tuple[Float[torch.Tensor, "B L head_dim"], Float[torch.Tensor, "B L head_dim"]]
        ] = None,
    ) -> Tuple[Float[torch.Tensor, "B L d"], Float[torch.Tensor, "B H L L"]]:
        """Forward pass for StandardDecoderLayer.

        Args:
            hidden_states (Float[torch.Tensor, "B L d"]): Input tensor of shape (batch_size, seq_len, hidden_size).
            attention_mask (Optional[Float[torch.Tensor, "B 1 L L"]]): Optional attention mask added to the
                attention scores before softmax, where the masked positions are indicated by large negative values.
            position_embeddings (Optional[Tuple[Float[torch.Tensor, "B L head_dim"], Float[torch.Tensor, "B L head_dim"]]]):
                Optional tuple of cosine and sine embeddings for RoPE.

        Returns:
            Tuple[Float[torch.Tensor, "B L d"], Float[torch.Tensor, "B H L L"]]: A tuple containing the output
                tensor and the attention weights tensor.
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)

        return hidden_states, attn_weights


class SequentialTransductionUnit(nn.Module):
    """Sequential Transduction Unit (STU) layer.

    Compared to standard STU layer, this module provides several options to generalize and
    enhance the performance of HSTU, including:

    - Option to switch the original learnable relative positional embeddings with Rotary Positional
        Embeddings (RoPE) for better extrapolation to longer sequences and improved performance.
    - Option to disable the original attention gating mechanism, allowing for a standard attention
        computation, which can be beneficial in certain scenarios where gating may not be stable.
    - Option to restore the FFN after attention, which was removed in the original STU design, to enhance
        the model's capacity.

    References:
        - Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for
            Generative Recommendations. ICML '24.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        attention_dropout: float = 0.0,
        attention_bias: bool = False,
        linear_dropout: float = 0.0,
        ffn_bias: bool = False,
        max_seq_len: int = 512,
        num_buckets: int = 128,
        enable_learnable_rel_posemb: bool = True,
        enable_attention_gating: bool = True,
        enable_ffn: bool = False,
    ) -> None:
        """Initializes SequentialTransductionUnit module.

        Args:
            hidden_size (int): Dimensionality of the model's hidden representations.
            num_heads (int): Number of attention heads.
            intermediate_size (int): Dimensionality of the feed-forward network's intermediate layer.
                Note that this is only used if `enable_ffn` is True.
            attention_dropout (float): Dropout rate for attention weights. Default is 0.0.
            attention_bias (bool): Whether to include bias terms in the attention projections. Default is False.
            linear_dropout (float): Dropout rate for the attention output before the final output projection, which
                is applied in `av_output` when attention gating is enabled. Note that this is only used if
                `enable_attention_gating` is True. Default is 0.0.
            ffn_bias (bool): Whether to include bias terms in the feed-forward network projections.
                Note that this is only used if `enable_ffn` is True. Default is False.
            max_seq_len (int): Maximum sequence length for relative positional embeddings. Default is 512.
            num_buckets (int): Number of buckets for relative positional embeddings. Default is 128.
            enable_learnable_rel_posemb (bool): Whether to use learnable relative positional embeddings.
                If False, RoPE will be used instead. Default is True.
            enable_attention_gating (bool): Whether to enable the attention gating mechanism. If False,
                standard attention computation is used. Default is True.
            enable_ffn (bool): Whether to include a feed-forward network after attention. Default is False.
        """
        super().__init__()

        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads."
        self.head_dim = hidden_size // num_heads
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.attention_dropout = attention_dropout
        self.attention_bias = attention_bias
        self.linear_dropout = linear_dropout
        self.ffn_bias = ffn_bias
        self.max_seq_len = max_seq_len
        self.num_buckets = num_buckets
        self.enable_learnable_rel_posemb = enable_learnable_rel_posemb
        self.enable_attention_gating = enable_attention_gating
        self.enable_ffn = enable_ffn

        self.self_attn = MaskedHSTUAttention(
            hidden_size=hidden_size,
            head_dim=self.head_dim,
            num_heads=num_heads,
            attention_dropout=attention_dropout,
            attention_bias=attention_bias,
            linear_dropout=linear_dropout,
            max_seq_len=max_seq_len,
            num_buckets=num_buckets,
            enable_learnable_rel_posemb=enable_learnable_rel_posemb,
            enable_attention_gating=enable_attention_gating,
        )
        self.input_layernorm = RMSNorm(hidden_size)

        self.mlp: Optional[SwiGLU] = None
        self.post_attention_layernorm: Optional[RMSNorm] = None
        if self.enable_ffn:
            self.mlp = SwiGLU(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                ffn_bias=ffn_bias,
            )
            self.post_attention_layernorm = RMSNorm(hidden_size)

    def forward(
        self,
        hidden_states: Float[torch.Tensor, "B L d"],
        attention_mask: Optional[Float[torch.Tensor, "B 1 L L"]] = None,
        position_embeddings: Optional[
            Tuple[Float[torch.Tensor, "B L head_dim"], Float[torch.Tensor, "B L head_dim"]]
        ] = None,
        timestamps: Optional[Int[torch.Tensor, "B L"]] = None,
    ) -> Tuple[Float[torch.Tensor, "B L d"], Float[torch.Tensor, "B H L L"]]:
        """Forward pass for SequentialTransductionUnit.

        Args:
            hidden_states (Float[torch.Tensor, "B L d"]): Input tensor of shape (batch_size, seq_len, hidden_size).
            attention_mask (Optional[Float[torch.Tensor, "B 1 L L"]]): Optional attention mask added to the attention
                scores before silu attention, where the masked positions are indicated by large negative values.
            position_embeddings (Optional[Tuple[Float[torch.Tensor, "B L head_dim"], Float[torch.Tensor, "B L head_dim"]]]):
                Optional tuple of cosine and sine embeddings for RoPE. Note that when `enable_learnable_rel_posemb` is True,
                this argument will be ignored.
            timestamps (Optional[Int[torch.Tensor, "B L"]]): Optional timestamps for each item in the sequence.

        Returns:
            Tuple[Float[torch.Tensor, "B L d"], Float[torch.Tensor, "B H L L"]]: A tuple containing the output
                tensor and the attention weights tensor.
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            timestamps=timestamps,
        )
        hidden_states = residual + hidden_states

        if self.enable_ffn and self.mlp is not None and self.post_attention_layernorm is not None:
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = residual + self.mlp(hidden_states)

        return hidden_states, attn_weights
