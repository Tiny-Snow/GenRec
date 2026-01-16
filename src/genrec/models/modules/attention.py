"""Multi-Head Attention modules."""

from __future__ import annotations

from typing import Optional, Tuple

from jaxtyping import Float, Int
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layernorm import RMSNorm
from .posemb import RelativeBucketedTimeAndPositionAttentionBias, apply_rotary_pos_emb

__all__ = [
    "MaskedSelfAttentionWithRoPE",
    "MaskedHSTUAttention",
]


class MaskedSelfAttentionWithRoPE(nn.Module):
    """Masked Multi-Head Self-Attention with RoPE, following `LlamaAttention`'s implementation.
    Note that we do not apply Grouped Query Attention (GQA) here. In addition, we do not
    support kv cache or flash attention in this implementation.

    .. note::
        This implementation calculates attention blocks manually to support attention weights
        output for interpretability and possibly other purposes (e.g., optimization).
    """

    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_heads: int,
        attention_dropout: float = 0.0,
        attention_bias: bool = False,
    ) -> None:
        """Initializes MaskedSelfAttentionWithRoPE module.

        Args:
            hidden_size (int): Dimensionality of the model's hidden representations.
            head_dim (int): Dimensionality of each attention head.
            num_heads (int): Number of attention heads.
            attention_dropout (float): Dropout rate for attention weights. Default is 0.0.
            attention_bias (bool): Whether to include bias terms in the attention projections. Default is False.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.attention_bias = attention_bias

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=attention_bias)

    # TODO: Optimize with flash attention if return attn_weights is False?
    # TODO: Implement kv cache for faster inference?
    # TODO: Implement grouped query attention (GQA)?
    def forward(
        self,
        hidden_states: Float[torch.Tensor, "B L d"],
        attention_mask: Optional[Float[torch.Tensor, "B 1 L L"]] = None,
        position_embeddings: Optional[
            Tuple[Float[torch.Tensor, "B L head_dim"], Float[torch.Tensor, "B L head_dim"]]
        ] = None,
    ) -> Tuple[Float[torch.Tensor, "B L d"], Float[torch.Tensor, "B H L L"]]:
        """Forward pass for MaskedSelfAttentionWithRoPE.

        Args:
            hidden_states (Float[torch.Tensor, "B L d"]): Input tensor of shape (batch_size, seq_len, hidden_size).
            attention_mask (Optional[Float[torch.Tensor, "B 1 L L"]]): Optional attention mask added to the scores
                before softmax.
            position_embeddings (Optional[Tuple[Float[torch.Tensor, "B L head_dim"], Float[torch.Tensor, "B L head_dim"]]]):
                Optional tuple of cosine and sine embeddings for RoPE.

        Returns:
            Tuple[Float[torch.Tensor, "B L d"], Float[torch.Tensor, "B H L L"]]: Output tensor and attention
                weights tensor.
        """
        batch_size, seq_len, _ = hidden_states.shape
        hidden_shape = (batch_size, seq_len, self.num_heads, self.head_dim)

        query_states: Float[torch.Tensor, "B H L head_dim"]
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states: Float[torch.Tensor, "B H L head_dim"]
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states: Float[torch.Tensor, "B H L head_dim"]
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        attn_weights: Float[torch.Tensor, "B H L L"]
        attn_weights = query_states @ key_states.transpose(-1, -2)
        attn_weights = attn_weights * (self.head_dim**-0.5)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_weights = attn_weights.to(query_states.dtype)

        softmax_output: Float[torch.Tensor, "B L H head_dim"]
        softmax_output = (attn_weights @ value_states).transpose(1, 2).contiguous()

        attn_output: Float[torch.Tensor, "B L d"]
        attn_output = self.o_proj(softmax_output.view(batch_size, seq_len, -1))

        return attn_output, attn_weights


class MaskedHSTUAttention(nn.Module):
    """Hierarchical Sequential Transduction Unit (HSTU) Attention module.

    Compared to standard HSTU attention, this module provides several options to generalize
    and enhance the attention mechanism, including:

    - Option to switch the original learnable relative positional embeddings with Rotary Positional
        Embeddings (RoPE) for better extrapolation to longer sequences and improved performance.
    - Option to disable the original attention gating mechanism, allowing for a standard attention
        computation, which can be beneficial in certain scenarios where gating may not be stable.

    References:
        - Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for
            Generative Recommendations. ICML '24.
    """

    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_heads: int,
        attention_dropout: float = 0.0,
        attention_bias: bool = False,
        linear_dropout: float = 0.0,
        max_seq_len: int = 512,
        num_buckets: int = 128,
        enable_learnable_rel_posemb: bool = True,
        enable_attention_gating: bool = True,
    ) -> None:
        """Initializes MaskedHSTUAttention module.

        Args:
            hidden_size (int): Dimensionality of the model's hidden representations.
            head_dim (int): Dimensionality of each attention head.
            num_heads (int): Number of attention heads.
            attention_dropout (float): Dropout rate for attention weights. Default is 0.0.
            attention_bias (bool): Whether to include bias terms in the attention projections. Default is False.
            linear_dropout (float): Dropout rate for the attention output before the final output projection, which
                is applied in `av_output` when attention gating is enabled. Note that when attention gating is disabled,
                this dropout will not be applied. Default is 0.0.
            max_seq_len (int): Maximum sequence length for relative positional embeddings. Default is 512.
            num_buckets (int): Number of buckets for relative positional embeddings. Default is 128.
            enable_learnable_rel_posemb (bool): Whether to use learnable relative positional embeddings.
                If False, RoPE will be used instead. Default is True.
            enable_attention_gating (bool): Whether to enable the attention gating mechanism. If False,
                standard attention computation is used. Default is True.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.attention_bias = attention_bias
        self.linear_dropout = linear_dropout
        self.max_seq_len = max_seq_len
        self.num_buckets = num_buckets
        self.enable_learnable_rel_posemb = enable_learnable_rel_posemb
        self.enable_attention_gating = enable_attention_gating

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=attention_bias)

        self.u_proj: Optional[nn.Linear] = None
        self.av_output_layernorm: Optional[RMSNorm] = None
        if self.enable_attention_gating:
            self.u_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=attention_bias)
            self.av_output_layernorm = RMSNorm(num_heads * head_dim)

        self.rel_attn_bias: Optional[RelativeBucketedTimeAndPositionAttentionBias] = None
        if self.enable_learnable_rel_posemb:
            self.rel_attn_bias = RelativeBucketedTimeAndPositionAttentionBias(
                max_seq_len=max_seq_len,
                num_buckets=num_buckets,
                bucketization_fn=lambda x: (torch.log(x.abs().clamp(min=1).float()) / 0.301).long(),
            )  # log10(2) ≈ 0.301

    # TODO: Optimize with flash attention if return attn_weights is False?
    # TODO: Implement kv cache for faster inference?
    # TODO: Implement grouped query attention (GQA)?
    def forward(
        self,
        hidden_states: Float[torch.Tensor, "B L d"],
        attention_mask: Optional[Float[torch.Tensor, "B 1 L L"]] = None,
        position_embeddings: Optional[
            Tuple[Float[torch.Tensor, "B L head_dim"], Float[torch.Tensor, "B L head_dim"]]
        ] = None,
        timestamps: Optional[Int[torch.Tensor, "B L"]] = None,
    ) -> Tuple[Float[torch.Tensor, "B L d"], Float[torch.Tensor, "B H L L"]]:
        """Forward pass for MaskedHSTUAttention.

        Args:
            hidden_states (Float[torch.Tensor, "B L d"]): Input tensor of shape (batch_size, seq_len, hidden_size).
            attention_mask (Optional[Float[torch.Tensor, "B 1 L L"]]): Optional attention mask added to the scores
                before silu attention.
            position_embeddings (Optional[Tuple[Float[torch.Tensor, "B L head_dim"], Float[torch.Tensor, "B L head_dim"]]]):
                Optional tuple of cosine and sine embeddings for RoPE. Note that when `enable_learnable_rel_posemb` is True,
                this argument will be ignored.
            timestamps (Optional[Int[torch.Tensor, "B L"]]): Optional timestamps for each item in the sequence.

        Returns:
            Tuple[Float[torch.Tensor, "B L d"], Float[torch.Tensor, "B H L L"]]: Output tensor and attention
                weights tensor.
        """
        batch_size, seq_len, _ = hidden_states.shape
        hidden_shape = (batch_size, seq_len, self.num_heads, self.head_dim)

        query_states: Float[torch.Tensor, "B H L head_dim"]
        query_states = F.silu(self.q_proj(hidden_states)).view(hidden_shape).transpose(1, 2)
        key_states: Float[torch.Tensor, "B H L head_dim"]
        key_states = F.silu(self.k_proj(hidden_states)).view(hidden_shape).transpose(1, 2)
        value_states: Float[torch.Tensor, "B H L head_dim"]
        value_states = F.silu(self.v_proj(hidden_states)).view(hidden_shape).transpose(1, 2)

        # apply RoPE is not using learnable relative positional embeddings
        if position_embeddings is not None and not self.enable_learnable_rel_posemb:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        attn_weights: Float[torch.Tensor, "B H L L"]
        attn_weights = query_states @ key_states.transpose(-1, -2)

        # apply learnable relative positional embeddings if enabled and the timestamps are provided
        if self.enable_learnable_rel_posemb and timestamps is not None and self.rel_attn_bias is not None:
            attn_weights = attn_weights + self.rel_attn_bias(timestamps)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask  # -inf -> 0 after SiLU activation

        # HSTU applies SiLU activation to attention weights instead of softmax
        attn_weights = F.silu(attn_weights) / seq_len
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        av_output: Float[torch.Tensor, "B L H*head_dim"]
        av_output = (attn_weights @ value_states).transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        # HSTU GLU in attention
        if self.enable_attention_gating and self.u_proj is not None and self.av_output_layernorm is not None:
            attn_gate: Float[torch.Tensor, "B L H*head_dim"]
            attn_gate = F.silu(self.u_proj(hidden_states))
            av_output = attn_gate * self.av_output_layernorm(av_output)
            av_output = F.dropout(av_output, p=self.linear_dropout, training=self.training)

        attn_output: Float[torch.Tensor, "B L d"]
        attn_output = self.o_proj(av_output)

        return attn_output, attn_weights
