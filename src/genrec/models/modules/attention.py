"""Multi-Head Attention modules."""

from __future__ import annotations

from typing import Optional, Tuple, Union

from jaxtyping import Float, Int
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import Cache, EncoderDecoderCache

from .layernorm import RMSNorm
from .posemb import RelativeBucketedTimeAndPositionAttentionBias, T5RelativePositionBias, apply_rotary_pos_emb

__all__ = [
    "MaskedHSTUAttention",
    "MaskedSelfAttentionWithRoPE",
    "T5Attention",
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


class T5Attention(nn.Module):
    """Multi-Head Attention module used in T5 model, following `T5Attention`'s implementation.

    Compared to standard T5Attention, this module provides several options to generalize
    and enhance the attention mechanism, including:
    - Option to switch the original learnable relative attention bias with Rotary Positional
    Embeddings (RoPE) for better extrapolation to longer sequences and improved performance.

    References:
    - Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. JMLR '20.
    """

    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_heads: int,
        attention_dropout: float = 0.0,
        attention_bias: bool = False,
        is_decoder: bool = False,
        has_relative_attention_bias: bool = False,
        relative_attention_num_buckets: int = 32,
        relative_attention_max_distance: int = 128,
        enable_rope: bool = False,
        layer_idx: Optional[int] = None,
    ) -> None:
        """Initializes T5Attention module.

        Args:
            hidden_size (int): Dimensionality of the model's hidden representations.
            head_dim (int): Dimensionality of each attention head.
            num_heads (int): Number of attention heads.
            attention_dropout (float): Dropout rate for attention weights. Default is 0.0.
            attention_bias (bool): Whether to include bias terms in the attention projections. Default is False.
            is_decoder (bool): Whether this attention module is used in the decoder. This is used to determine
                the directionality of relative positional embeddings. Default is False.
            has_relative_attention_bias (bool): Whether to compute learnable relative positional bias. If False, this
                module will not initialize a `T5RelativePositionBias` instance. Typically, T5 set `has_relative_attention_bias`
                to True only for the first block, while the rest blocks reuse the same relative positional bias. Note
                that when `enable_rope` is True, this argument will be ignored. Default is False.
            relative_attention_num_buckets (int): Number of buckets for relative positional embeddings. Default is 32.
            relative_attention_max_distance (int): Maximum distance for relative positional embeddings. Default is 128.
            enable_rope (bool): Whether to use RoPE instead of learnable relative positional bias. If False, the original
                learnable relative positional bias in T5 will be used. Default is False.
            layer_idx (Optional[int]): Optional layer index of this attention module in the model. This should be set
                when caching past key/values in the decoder for autoregressive generation. Default is None.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.attention_bias = attention_bias
        self.is_decoder = is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.enable_rope = enable_rope
        self.layer_idx = layer_idx

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=attention_bias)

        self.rel_pos_bias: Optional[T5RelativePositionBias] = None
        if self.has_relative_attention_bias and not self.enable_rope:
            self.rel_pos_bias = T5RelativePositionBias(
                num_buckets=relative_attention_num_buckets,
                max_distance=relative_attention_max_distance,
                num_heads=num_heads,
                is_bidirectional=(not is_decoder),
            )

    def forward(
        self,
        hidden_states: Float[torch.Tensor, "B L_q d"],
        attention_mask: Optional[Float[torch.Tensor, "B 1 #L_q L_k"]] = None,
        key_value_states: Optional[Float[torch.Tensor, "B L_k d"]] = None,
        position_bias: Optional[Float[torch.Tensor, "#B H L_q L_k"]] = None,
        position_embeddings: Optional[
            Tuple[Float[torch.Tensor, "B L_q head_dim"], Float[torch.Tensor, "B L_q head_dim"]]
        ] = None,
        past_key_values: Optional[EncoderDecoderCache] = None,
        cache_position: Optional[Int[torch.Tensor, "L_q"]] = None,
        output_attentions: bool = False,
    ) -> Tuple[
        Float[torch.Tensor, "B L_q d"],
        Optional[Float[torch.Tensor, "#B H L_q L_k"]],
        Optional[Float[torch.Tensor, "B H L_q L_k"]],
    ]:
        """Forward pass for T5Attention.

        Args:
            hidden_states (Float[torch.Tensor, "B L_q d"]): Input tensor of shape (batch_size, query_len, hidden_size).
            attention_mask (Optional[Float[torch.Tensor, "B 1 #L_q L_k"]]): Optional attention mask added to the scores
                before softmax, with shape either (batch_size, 1, query_len, key_len) for causal self-attention in
                decoder, or (batch_size, 1, 1, key_len) for self-attention in encoder and cross-attention in decoder.
                Specifically, if the dimension `L_k` is longer than the actual key length (which can happen during
                autoregressive generation in the decoder), the mask will be sliced accordingly. Default is None.
            key_value_states (Optional[Float[torch.Tensor, "B L_k d"]]): Optional tensor for key and value states, which
                assumes to be the output of the encoder and is used in the decoder cross-attention. If None, self-attention
                is performed; otherwise, cross-attention is performed. Default is None.
            position_bias (Optional[Float[torch.Tensor, "#B H L_q L_k"]]): Optional precomputed position bias to be added
                to the attention scores. If None, and if `has_relative_attention_bias` is True and `enable_rope` is False,
                the position bias will be computed based on the relative positions of the queries and keys from the T5
                learnable relative positional embeddings. Default is None.
            position_embeddings (Optional[Tuple[Float[torch.Tensor, "B L_q head_dim"], Float[torch.Tensor, "B L_q head_dim"]]]):
                Optional tuple of cosine and sine embeddings for RoPE. Note that when `enable_rope` is False, this argument
                will be ignored. Default is None.
            past_key_values (Optional[EncoderDecoderCache]): Optional cache for previously computed key and value states,
                used in the decoder for faster autoregressive generation. Default is None.
            cache_position (Optional[Int[torch.Tensor, "L_q"]]): Optional position IDs used to compute relative positions
                when caching past key/values in the decoder. If provided, it should contain the absolute positions of the
                current query tokens. Default is None.
            output_attentions (bool): Whether to return the attention weights along with the output. Default is False.

        Returns:
            Tuple[
                Float[torch.Tensor, "B L_q d"],
                Optional[Float[torch.Tensor, "#B H L_q L_k"]],
                Optional[Float[torch.Tensor, "B H L_q L_k"]],
            ]: Output tensor, position bias, and attention weights tensor. If `output_attentions` is False, the attention weights
                will be None.
        """
        B, L_q, _ = hidden_states.shape
        H, head_dim = self.num_heads, self.head_dim

        # Compute query states
        query_states: Float[torch.Tensor, "B H L_q head_dim"]
        query_states = self.q_proj(hidden_states).view(B, L_q, H, head_dim).transpose(1, 2)

        # If key_value_states are provided this layer is used as a cross-attention layer for the decoder
        is_cross_attention = key_value_states is not None

        # Compute key and value states from either key_value_states or hidden_states
        is_updated = False  # indicates whether the cross-attention kv states are updated in the cache
        curr_past_key_value: Optional[Cache] = None
        if past_key_values is not None:
            assert self.layer_idx is not None, "layer_idx must not be None when accessing cached layers"
            is_updated = past_key_values.is_updated.get(self.layer_idx)
            if is_cross_attention:
                curr_past_key_value = past_key_values.cross_attention_cache
            else:
                curr_past_key_value = past_key_values.self_attention_cache

        # Compute key and value states, updating the cache if necessary
        key_states: Float[torch.Tensor, "B H L_k head_dim"]
        value_states: Float[torch.Tensor, "B H L_k head_dim"]
        current_states = key_value_states if is_cross_attention else hidden_states
        if is_cross_attention and past_key_values is not None and is_updated:
            # use cached key and value states for cross-attention
            assert curr_past_key_value is not None and self.layer_idx is not None
            key_states = curr_past_key_value.layers[self.layer_idx].keys  # type: ignore - lazy init in DynamicLayer
            value_states = curr_past_key_value.layers[self.layer_idx].values  # type: ignore - lazy init in DynamicLayer
        else:
            # cache miss for cross-attention, or in self-attention case
            key_states = self.k_proj(current_states).view(B, -1, H, head_dim).transpose(1, 2)
            value_states = self.v_proj(current_states).view(B, -1, H, head_dim).transpose(1, 2)

            # apply RoPE if not using learnable relative PE (only for self-attention, so L_q = L_k)
            if not is_cross_attention and position_embeddings is not None and self.enable_rope:
                cos, sin = position_embeddings
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            # update cache with new key and value states
            if past_key_values is not None:
                assert curr_past_key_value is not None and self.layer_idx is not None
                cache_position = cache_position if not is_cross_attention else None
                key_states, value_states = curr_past_key_value.update(
                    key_states, value_states, self.layer_idx, {"cache_position": cache_position}
                )
                if is_cross_attention:  # update cross-attention cache flag, as it is static after being set
                    past_key_values.is_updated[self.layer_idx] = True

        # Compute attention scores
        attn_weights: Float[torch.Tensor, "B H L_q L_k"]
        attn_weights = query_states @ key_states.transpose(-1, -2)

        # Apply relative position bias if enabled. When enabling RoPE, rel_pos_bias is ignored.
        L_k = key_states.size(-2)
        if position_bias is None:
            if self.rel_pos_bias is not None:
                real_L_q = cache_position[-1] + 1 if cache_position is not None else L_q
                position_bias = self.rel_pos_bias(
                    real_L_q, L_k, cache_position=cache_position, device=hidden_states.device
                )[:, :, -L_q:, :]
            else:
                position_bias = torch.zeros((1, H, L_q, L_k), device=hidden_states.device)

        # Apply attention mask, note that the returned position bias will contain the mask if provided
        assert position_bias is not None
        if attention_mask is not None:  # slice for longer attention mask
            causal_mask = attention_mask[:, :, :, :L_k]
            position_bias = position_bias + causal_mask

        attn_weights = attn_weights + position_bias

        # Compute attention scores and attention output
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        softmax_output: Float[torch.Tensor, "B L_q H head_dim"]
        softmax_output = (attn_weights @ value_states).transpose(1, 2).contiguous()

        attn_output: Float[torch.Tensor, "B L_q d"]
        attn_output = self.o_proj(softmax_output.view(B, L_q, -1))

        return attn_output, position_bias, attn_weights if output_attentions else None
