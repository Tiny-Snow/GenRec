"""Standard layers."""

from __future__ import annotations

from typing import Optional, Tuple

from jaxtyping import Bool, Float, Int
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import (
    GatedMaskedSelfAttentionWithRoPE,
    GatedMaskedSelfAttentionWithRoPEAndSiLUActivation,
    MaskedSelfAttentionWithRoPE,
    MaskedSelfAttentionWithRoPEAndSiLUActivation,
)
from .feedforward import SwiGLU
from .layernorm import RMSNorm
from .posemb import RelativeBucketedTimeAndPositionAttentionBias

__all__ = [
    "LlamaDecoderLayer",
    "LlamaDecoder2HSTULayer",
    "SequentialTransductionUnit",
]


class LlamaDecoder2HSTULayer(nn.Module):
    """A standard Llama Transformer Decoder Layer -> HSTU Layer, following
    `transformers.LlamaDecoderLayer`'s implementation."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        attention_dropout: float = 0.0,
        attention_bias: bool = False,
        ffn_bias: bool = False,
        attention_type: str = "softmax",
    ) -> None:
        """Initializes StandardDecoderLayer module.

        Args:
            hidden_size (int): Dimensionality of the model's hidden representations.
            num_heads (int): Number of attention heads.
            intermediate_size (int): Dimensionality of the feed-forward network's intermediate layer.
            attention_dropout (float): Dropout rate for attention weights. Default is 0.0.
            attention_bias (bool): Whether to include bias terms in the attention projections. Default is False.
            ffn_bias (bool): Whether to include bias terms in the feed-forward network projections. Default is False.
            attention_type (str): Type of attention normalization to use ("softmax", "silu", "silu_norm").
                Default is "softmax".
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
        self.attention_type = attention_type

        if self.attention_type == "softmax":
            self.self_attn = MaskedSelfAttentionWithRoPE(
                hidden_size=hidden_size,
                head_dim=self.head_dim,
                num_heads=num_heads,
                attention_dropout=attention_dropout,
                attention_bias=attention_bias,
            )
        elif self.attention_type == "silu":
            self.self_attn = MaskedSelfAttentionWithRoPEAndSiLUActivation(
                hidden_size=hidden_size,
                head_dim=self.head_dim,
                num_heads=num_heads,
                attention_dropout=attention_dropout,
                attention_bias=attention_bias,
                attention_norm=False,
            )
        elif self.attention_type == "silu_norm":
            self.self_attn = MaskedSelfAttentionWithRoPEAndSiLUActivation(
                hidden_size=hidden_size,
                head_dim=self.head_dim,
                num_heads=num_heads,
                attention_dropout=attention_dropout,
                attention_bias=attention_bias,
                attention_norm=True,
            )
        elif self.attention_type == "gated_softmax":
            self.self_attn = GatedMaskedSelfAttentionWithRoPE(
                hidden_size=hidden_size,
                head_dim=self.head_dim,
                num_heads=num_heads,
                attention_dropout=attention_dropout,
                attention_bias=attention_bias,
            )
        elif self.attention_type == "gated_silu":
            self.self_attn = GatedMaskedSelfAttentionWithRoPEAndSiLUActivation(
                hidden_size=hidden_size,
                head_dim=self.head_dim,
                num_heads=num_heads,
                attention_dropout=attention_dropout,
                linear_dropout=0.0,
                attention_bias=attention_bias,
                attention_norm=False,
            )
        elif self.attention_type == "gated_silu_norm":
            self.self_attn = GatedMaskedSelfAttentionWithRoPEAndSiLUActivation(
                hidden_size=hidden_size,
                head_dim=self.head_dim,
                num_heads=num_heads,
                attention_dropout=attention_dropout,
                linear_dropout=0.0,
                attention_bias=attention_bias,
                attention_norm=True,
            )
        else:
            raise ValueError(f"Unsupported attention_type: {self.attention_type}")
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
        add_ffn: bool = False,
        softmax_attention: bool = False,
        attention_norm: bool = False,
        time_interval: float = 1.0,
        relative_position_bias: bool = True,
        attention_gating: bool = True,
    ) -> None:
        """Initializes SequentialTransductionUnit module.

        Args:
            hidden_size (int): Dimensionality of the model's hidden representations.
            num_heads (int): Number of attention heads.
            max_seq_len (int): Maximum sequence length for relative position embeddings.
            num_buckets (int): Number of buckets for relative position bucketization. Default is 128.
            linear_dropout (float): Dropout rate for linear layers. Default is 0.0.
            attention_dropout (float): Dropout rate for attention weights. Default is 0.0.
            add_ffn (bool): Whether to add feed-forward network after attention. Default is False.
            softmax_attention (bool): Whether to use softmax-based attention mechanism rather than
                the original silu-based attention mechanism. Default is False.
            attention_norm (bool): Whether to apply row-wise normalization to attention scores.
                Default is False.
            time_interval (float): Factor to divide Unix timestamps by before bucketization. Default is 1.0
                (seconds). Use larger values (e.g., 86400) to operate on coarser units such as days.
            relative_position_bias (bool): Whether to use relative position bias. Default is True.
            attention_gating (bool): Whether to use attention gating mechanism. Default is True.
        """
        super().__init__()

        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads."
        self.head_dim = hidden_size // num_heads
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.linear_dropout = linear_dropout
        self.attention_dropout = attention_dropout
        self.add_ffn = add_ffn
        self.softmax_attention = softmax_attention
        self.attention_norm = attention_norm
        self.time_interval = float(time_interval)
        if self.time_interval <= 0:
            raise ValueError("time_interval must be positive.")
        self.relative_position_bias = relative_position_bias
        self.attention_gating = attention_gating

        self.input_layernorm = RMSNorm(hidden_size)
        self.attn_output_layernorm: Optional[RMSNorm] = None
        if self.attention_gating:
            self.attn_output_layernorm = RMSNorm(hidden_size)

        self.u_proj: Optional[nn.Sequential] = None
        if self.attention_gating:
            self.u_proj = nn.Sequential(nn.Linear(hidden_size, hidden_size, bias=False), nn.SiLU())
        self.v_proj = nn.Sequential(nn.Linear(hidden_size, hidden_size, bias=False), nn.SiLU())
        self.q_proj = nn.Sequential(nn.Linear(hidden_size, hidden_size, bias=False), nn.SiLU())
        self.k_proj = nn.Sequential(nn.Linear(hidden_size, hidden_size, bias=False), nn.SiLU())
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.mlp: Optional[SwiGLU] = None
        self.post_attention_layernorm: Optional[RMSNorm] = None
        if self.add_ffn:
            self.mlp = SwiGLU(
                hidden_size=hidden_size,
                intermediate_size=4 * hidden_size,
                ffn_bias=False,
            )
            self.post_attention_layernorm = RMSNorm(hidden_size)

        self.rel_attn_bias: Optional[RelativeBucketedTimeAndPositionAttentionBias] = None
        if self.relative_position_bias:
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
        if timestamps is not None and self.rel_attn_bias is not None:
            scaled_timestamps = self._scale_timestamps(timestamps)
            qk_attn = qk_attn + self.rel_attn_bias(scaled_timestamps)
        if self.softmax_attention:
            qk_attn = F.softmax(qk_attn / (hd**0.5), dim=-1, dtype=torch.float32).to(qk_attn.dtype)
        else:
            qk_attn = F.silu(qk_attn) / L

        if self.attention_norm:
            qk_attn = qk_attn / (qk_attn.sum(dim=-1, keepdim=True) + 1e-8)

        if attention_mask is not None:
            qk_attn = qk_attn * attention_mask.to(qk_attn.dtype)
        qk_attn = F.dropout(qk_attn, p=self.attention_dropout, training=self.training)

        attn_output: Float[torch.Tensor, "B L d"] = (qk_attn @ v).transpose(1, 2).contiguous().view(B, L, d)

        # HSTU GLU
        if self.attention_gating and self.u_proj is not None and self.attn_output_layernorm is not None:
            u: Float[torch.Tensor, "B L d"] = self.u_proj(hidden_states)
            hidden_states = u * self.attn_output_layernorm(attn_output)
            hidden_states = F.dropout(hidden_states, p=self.linear_dropout, training=self.training)
        hidden_states = self.o_proj(hidden_states)

        hidden_states = residual + hidden_states

        # Optional feed-forward network
        if self.add_ffn and self.mlp is not None and self.post_attention_layernorm is not None:
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = residual + self.mlp(hidden_states)

        return hidden_states, qk_attn

    def _scale_timestamps(
        self,
        timestamps: Int[torch.Tensor, "B L"],
    ) -> Int[torch.Tensor, "B L"]:
        """Scales raw Unix timestamps by the configured time interval before bucketization."""
        if self.time_interval == 1.0:
            return timestamps

        scaled: Float[torch.Tensor, "B L"]
        scaled = torch.floor(timestamps.to(torch.float32) / self.time_interval)
        return scaled.to(torch.long)
