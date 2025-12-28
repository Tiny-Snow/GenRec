"""Utilities for modules."""

from __future__ import annotations

from typing import Optional

import torch
from jaxtyping import Float

__all__ = [
    "create_attention_mask",
]


def create_attention_mask(
    attention_mask: Float[torch.Tensor, "B seq_len"],
    tgt_len: Optional[int] = None,
    is_causal: bool = True,
    dtype: torch.dtype = torch.float32,
) -> Float[torch.Tensor, "B 1 tgt_len seq_len"]:
    """Creates a 4D attention mask from a 2D attention mask.

    Args:
        attention_mask (Float[torch.Tensor, "B seq_len"]): 2D mask where 1 indicates valid tokens and
            0 indicates padding tokens.
        tgt_len (Optional[int]): Target sequence length. Defaults to `seq_len` when None.
        is_causal (bool): Whether to apply causal masking. Default is True.
        dtype (torch.dtype): Data type of the output mask. Default is `torch.float32`.

    Returns:
        Float[torch.Tensor, "B 1 tgt_len seq_len"]: Attention mask tensor where masked positions
            are set to `-inf` and unmasked positions are set to 0.
    """
    batch_size, seq_len = attention_mask.shape
    tgt_len = tgt_len if tgt_len is not None else seq_len

    expanded_mask = attention_mask[:, None, None, :].expand(batch_size, 1, tgt_len, seq_len)
    float_mask = expanded_mask.to(dtype=dtype)
    float_mask = (1.0 - float_mask) * torch.finfo(dtype).min

    if is_causal:
        causal_mask = torch.tril(torch.ones((tgt_len, seq_len), device=attention_mask.device, dtype=dtype))
        causal_mask = causal_mask[None, None, :, :]
        float_mask = float_mask + (1.0 - causal_mask) * torch.finfo(dtype).min

    return float_mask
