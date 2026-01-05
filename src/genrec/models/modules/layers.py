"""Standard layers."""

from __future__ import annotations

from typing import Optional, Tuple

from jaxtyping import Bool, Float, Int
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import MaskedSelfAttentionWithRoPE
from .feedforward import SwiGLU
from .layernorm import RMSNorm
from .posemb import RelativeBucketedTimeAndPositionAttentionBias

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
    """Sequential Transduction Unit (STU) layer as described in the HSTU paper.

    References:
        - Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for
            Generative Recommendations. ICML '24.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        max_seq_len: int,
        num_buckets: int = 128,
        linear_dropout: float = 0.0,
        attention_dropout: float = 0.0,
    ) -> None:
        """Initializes SequentialTransductionUnit module.

        Args:
            hidden_size (int): Dimensionality of the model's hidden representations.
            num_heads (int): Number of attention heads.
            max_seq_len (int): Maximum sequence length for relative position embeddings.
            linear_dropout (float): Dropout rate for linear layers. Default is 0.0.
            attention_dropout (float): Dropout rate for attention weights. Default is 0.0.
        """
        super().__init__()

        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads."
        self.head_dim = hidden_size // num_heads
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.linear_dropout = linear_dropout
        self.attention_dropout = attention_dropout

        self.input_layernorm = RMSNorm(hidden_size)
        self.attn_output_layernorm = RMSNorm(hidden_size)

        self.u_proj = nn.Sequential(nn.Linear(hidden_size, hidden_size, bias=False), nn.SiLU())
        self.v_proj = nn.Sequential(nn.Linear(hidden_size, hidden_size, bias=False), nn.SiLU())
        self.q_proj = nn.Sequential(nn.Linear(hidden_size, hidden_size, bias=False), nn.SiLU())
        self.k_proj = nn.Sequential(nn.Linear(hidden_size, hidden_size, bias=False), nn.SiLU())
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.rel_attn_bias = RelativeBucketedTimeAndPositionAttentionBias(
            max_seq_len=max_seq_len,
            num_buckets=num_buckets,
            bucketization_fn=lambda x: (torch.log(x.abs().clamp(min=1).float()) / 0.301).long(),
        )  # log10(2) ≈ 0.301

    def forward(
        self,
        hidden_states: Float[torch.Tensor, "B L d"],
        attention_mask: Optional[Bool[torch.Tensor, "B 1 L L"]] = None,
        timestamps: Optional[Int[torch.Tensor, "B L"]] = None,
    ) -> Tuple[Float[torch.Tensor, "B L d"], Float[torch.Tensor, "B H L L"]]:
        """Forward pass for SequentialTransductionUnit.

        Args:
            hidden_states (Float[torch.Tensor, "B L d"]): Input tensor of shape (batch_size, seq_len, hidden_size).
            attention_mask (Optional[Bool[torch.Tensor, "B 1 L L"]]): Optional attention mask where True indicates
                valid positions and False indicates masked positions.
            timestamps (Optional[Int[torch.Tensor, "B L"]]): Optional timestamps for each item in the sequence.

        Returns:
            Tuple[Float[torch.Tensor, "B L d"], Float[torch.Tensor, "B H L L"]]: A tuple containing the output
                tensor and the attention weights tensor.
        """
        B, L, d = hidden_states.shape
        H, hd = self.num_heads, self.head_dim

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # HSTU attention
        q: Float[torch.Tensor, "B H L head_dim"] = self.q_proj(hidden_states).view(B, L, H, hd).transpose(1, 2)
        k: Float[torch.Tensor, "B H L head_dim"] = self.k_proj(hidden_states).view(B, L, H, hd).transpose(1, 2)
        v: Float[torch.Tensor, "B H L head_dim"] = self.v_proj(hidden_states).view(B, L, H, hd).transpose(1, 2)

        qk_attn: Float[torch.Tensor, "B H L L"] = q @ k.transpose(-2, -1)
        if timestamps is not None:
            qk_attn = qk_attn + self.rel_attn_bias(timestamps)
        qk_attn = F.silu(qk_attn) / L

        if attention_mask is not None:
            qk_attn = qk_attn * attention_mask.to(qk_attn.dtype)
        qk_attn = F.dropout(qk_attn, p=self.attention_dropout, training=self.training)

        attn_output: Float[torch.Tensor, "B L d"] = (qk_attn @ v).transpose(1, 2).contiguous().view(B, L, d)

        # HSTU GLU
        u: Float[torch.Tensor, "B L d"] = self.u_proj(hidden_states)
        hidden_states = u * self.attn_output_layernorm(attn_output)
        hidden_states = F.dropout(hidden_states, p=self.linear_dropout, training=self.training)
        hidden_states = self.o_proj(hidden_states)

        hidden_states = residual + hidden_states
        return hidden_states, qk_attn
