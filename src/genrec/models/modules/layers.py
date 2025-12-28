"""Standard layers."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from jaxtyping import Float, Int

from .attention import MaskedSelfAttentionWithRoPE
from .feedforward import SwiGLU
from .layernorm import RMSNorm

__all__ = [
    "LlamaDecoderLayer",
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
                attention scores before softmax.
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
