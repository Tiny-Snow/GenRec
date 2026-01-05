"""Positional Encoding (PE) modules."""

from __future__ import annotations

from typing import Optional, Callable, Tuple

from jaxtyping import Float, Int
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "LearnableInputPositionalEmbedding",
    "RelativeBucketedTimeAndPositionAttentionBias",
    "RotaryEmbedding",
    "apply_rotary_pos_emb",
]


class LearnableInputPositionalEmbedding(nn.Module):
    """Learnable Input Positional Embedding module."""

    def __init__(
        self,
        max_position_embeddings: int,
        embed_dim: int,
        dropout_rate: float = 0.0,
    ) -> None:
        """Initializes LearnableInputPositionalEmbedding module.

        Args:
            max_position_embeddings (int): Maximum number of position embeddings.
            embed_dim (int): Dimensionality of the embeddings.
            dropout_rate (float): Dropout rate applied to the output embeddings. Default is 0.0.
        """
        super().__init__()
        self.pos_emb = nn.Embedding(max_position_embeddings, embed_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        x: Float[torch.Tensor, "B L d"],
        position_ids: Optional[Float[torch.Tensor, "B L"]] = None,
    ) -> Float[torch.Tensor, "B L d"]:
        """Applies learnable positional embeddings to the input tensor.

        Args:
            x (Float[torch.Tensor, "B L d"]): Input tensor of shape (batch_size, seq_len, embed_dim).
            position_ids (Optional[Float[torch.Tensor, "B L"]]): Optional position IDs used to index
                positional embeddings. If omitted, positions are assumed sequential from zero.

        Returns:
            Float[torch.Tensor, "B L d"]: Input tensor augmented with positional embeddings.
        """
        batch_size, seq_len, _ = x.shape

        # NOTE: by default, position_ids is created from right to left: [L-1, L-2, ..., 0].
        # This is to align with the left padding scheme.
        if position_ids is None:
            position_ids = torch.arange(seq_len - 1, -1, -1, device=x.device).unsqueeze(0).expand(batch_size, -1)

        pos_embeddings: Float[torch.Tensor, "B L d"]
        pos_embeddings = self.pos_emb(position_ids)

        # NOTE: here we do not scale the input embeddings by sqrt(d) before adding positional embeddings,
        # as the model initialization in PreTrainedModel's `init_weights` already takes care of proper scaling.
        x = x + pos_embeddings
        x = self.dropout(x)

        return x


class RelativeBucketedTimeAndPositionAttentionBias(nn.Module):
    """Relative Bucketed Time and Position Attention Bias for attention mechanism.

    This module computes a bias matrix based on the relative time differences
    and positional differences between elements in a sequence. The relative time
    differences are bucketed into discrete intervals to capture temporal relationships
    effectively. Note that the relative time in (i, j) position is computed as
    timestamp(j+1) - timestamp(i), following the official HSTU implementation. This
    can better capture the time gap between the current item and the next item,
    facilitating next-item prediction tasks.

    References:
        - Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for
            Generative Recommendations. ICML '24.
    """

    def __init__(
        self,
        max_seq_len: int,
        num_buckets: int,
        bucketization_fn: Callable[[Int[torch.Tensor, "..."]], Int[torch.Tensor, "..."]],
    ) -> None:
        """Initializes RelativeBucketedTimeAndPositionAttentionBias module.

        Args:
            max_seq_len (int): Maximum sequence length.
            num_buckets (int): Number of buckets for time differences.
            bucketization_fn (Callable[Int[torch.Tensor, "..."], Int[torch.Tensor, "..."]]):
                Function to bucketize time differences.
        """
        super().__init__()
        self.max_seq_len = max_seq_len
        self.num_buckets = num_buckets
        self.bucketization_fn = bucketization_fn

        # NOTE: the attention bias is shared across all heads
        self.time_bias_table = nn.Embedding(num_buckets + 1, 1)
        self.pos_bias_table = nn.Embedding(2 * max_seq_len - 1, 1)

    def forward(
        self,
        timestamps: Int[torch.Tensor, "B L"],
    ) -> Float[torch.Tensor, "B 1 L L"]:
        """Computes the relative bucketed time and position attention bias.

        Args:
            timestamps (Int[torch.Tensor, "B L"]): Timestamps of shape (batch_size, seq_len).

        Returns:
            Float[torch.Tensor, "B 1 L L"]: Attention bias tensor of shape (batch_size, 1, seq_len, seq_len).
                Note that the second dimension is 1 since the bias is shared across all heads.
        """
        B, L = timestamps.shape

        # compute relative positional differences
        pos_ids: Int[torch.Tensor, "L"] = torch.arange(L, device=timestamps.device)
        rel_pos_ids: Int[torch.Tensor, "L L"] = pos_ids[None, :] - pos_ids[:, None]
        rel_pos_ids_shifted: Int[torch.Tensor, "L L"] = rel_pos_ids + (self.max_seq_len - 1)
        rel_pos_bias: Float[torch.Tensor, "L L"] = self.pos_bias_table(rel_pos_ids_shifted).squeeze(-1)

        # compute relative time differences using ts(next) - ts(current) as in official HSTU
        ext_timestamps = torch.cat([timestamps, timestamps[:, -1:].expand(-1, 1)], dim=1)
        time_diffs: Int[torch.Tensor, "B L L"]
        time_diffs = ext_timestamps[:, 1:].unsqueeze(2) - ext_timestamps[:, :-1].unsqueeze(1)
        bucketed_time_diffs: Int[torch.Tensor, "B L L"] = self.bucketization_fn(time_diffs)
        bucketed_time_diffs = torch.clamp(bucketed_time_diffs, min=0, max=self.num_buckets).detach()
        rel_time_bias: Float[torch.Tensor, "B L L"] = self.time_bias_table(bucketed_time_diffs).squeeze(-1)

        return (rel_pos_bias.unsqueeze(0) + rel_time_bias).unsqueeze(1)


class RotaryEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE), following `LlamaRotaryEmbedding` implementation."""

    inv_freq: Float[torch.Tensor, "head_dim//2"]

    def __init__(
        self,
        head_dim: int,
        rope_theta: float = 10000.0,
        device=None,
    ) -> None:
        """Initializes RotaryEmbedding module.

        Args:
            head_dim (int): Dimensionality of each attention head.
            rope_theta (float): Base frequency for rotary embeddings. Default is 10000.0.
            device (Optional[str]): Device identifier used to store buffers.
        """
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for rotary embeddings."
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        inv_freq: Float[torch.Tensor, "head_dim//2"]
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(
        self,
        x: Float[torch.Tensor, "B L d"],
        position_ids: Optional[Float[torch.Tensor, "B L"]] = None,
    ) -> Tuple[Float[torch.Tensor, "B L head_dim"], Float[torch.Tensor, "B L head_dim"]]:
        """Generates rotary embeddings for input tensor.

        Args:
            x (Float[torch.Tensor, "B L head_dim"]): Input tensor of shape (batch_size, seq_len, hidden_size).
            position_ids (Optional[Float[torch.Tensor, "B L"]]): Optional position IDs used to compute
                rotary embeddings. If omitted, positions are assumed sequential from zero.

        Returns:
            Tuple[Float[torch.Tensor, "B L head_dim"], Float[torch.Tensor, "B L head_dim"]]: Cosine and sine
                embeddings corresponding to each position.

        .. note::
            The returned cosine and sine embeddings correspond to the dimensions (0, 2, 4, ...,
            head_dim-2, 1, 3, 5, ..., head_dim-1), which assumes interleaved arrangement for applying
            rotary embeddings. This slightly differs from the original RoPE formulation. However, given
            the symmetric nature of hidden dimensions, this arrangement works equivalently with the
            standard RoPE, while being more efficient to compute.
        """
        batch_size, seq_len, _ = x.shape
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)

        inv_freq_expanded: Float[torch.Tensor, "B head_dim//2 1"]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(batch_size, -1, 1).to(x.device)

        position_ids_expanded: Float[torch.Tensor, "B 1 L"]
        position_ids_expanded = position_ids[:, None, :].float()

        with torch.autocast(x.device.type, enabled=False):  # force FP32
            freqs: Float[torch.Tensor, "B L head_dim//2"]
            freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)

            emb: Float[torch.Tensor, "B L head_dim"]
            emb = torch.cat((freqs, freqs), dim=-1)

            cos, sin = emb.cos(), emb.sin()

        return cos, sin


def rotate_half(x: Float[torch.Tensor, "... dim"]) -> Float[torch.Tensor, "... dim"]:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    query: Float[torch.Tensor, "B H L head_dim"],
    key: Float[torch.Tensor, "B H L head_dim"],
    cos_emb: Float[torch.Tensor, "B L head_dim"],
    sin_emb: Float[torch.Tensor, "B L head_dim"],
) -> Tuple[Float[torch.Tensor, "B H L head_dim"], Float[torch.Tensor, "B H L head_dim"]]:
    """Applies rotary positional embeddings to query and key tensors.

    Args:
        query (Float[torch.Tensor, "B H L head_dim"]): Query tensor.
        key (Float[torch.Tensor, "B H L head_dim"]): Key tensor.
        cos_emb (Float[torch.Tensor, "B L head_dim"]): Cosine embeddings.
        sin_emb (Float[torch.Tensor, "B L head_dim"]): Sine embeddings.

    Returns:
        Tuple[Float[torch.Tensor, "B H L head_dim"], Float[torch.Tensor, "B H L head_dim"]]: Query and key
            tensors after applying rotary embeddings.
    """
    cos: Float[torch.Tensor, "B 1 L head_dim"] = cos_emb.unsqueeze(1)
    sin: Float[torch.Tensor, "B 1 L head_dim"] = sin_emb.unsqueeze(1)
    query_rotated = (query * cos) + (rotate_half(query) * sin)
    key_rotated = (key * cos) + (rotate_half(key) * sin)
    return query_rotated, key_rotated
