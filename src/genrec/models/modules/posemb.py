"""Positional Encoding (PE) modules."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from jaxtyping import Float

__all__ = [
    "RotaryEmbedding",
    "apply_rotary_pos_emb",
]


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
