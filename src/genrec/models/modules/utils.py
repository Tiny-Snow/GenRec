"""Utilities for modules."""

from __future__ import annotations

from typing import Optional, Union

from jaxtyping import Float, Int
import torch

__all__ = [
    "create_attention_mask",
]


def create_attention_mask(
    attention_mask: Int[torch.Tensor, "B seq_len"],
    tgt_len: Optional[int] = None,
    is_causal: bool = True,
    mask_value: Optional[float] = None,
    dtype: torch.dtype = torch.float32,
    cache_position: Optional[Union[Int[torch.Tensor, "#B seq_len"], Int[torch.Tensor, "seq_len"]]] = None,
    past_key_values_length: int = 0,
    kv_seq_len: Optional[int] = None,
) -> Float[torch.Tensor, "B 1 tgt_len key_len"]:
    """Creates a 4D attention mask from a 2D attention mask.

    Args:
        attention_mask (Int[torch.Tensor, "B seq_len"]): 2D mask where 1 indicates valid tokens and
            0 indicates padding tokens.
        tgt_len (Optional[int]): Target sequence length. Defaults to `seq_len` when None.
        is_causal (bool): Whether to apply causal masking. Default is True.
        mask_value (Optional[float]): Value to use for masked positions. If None, uses the minimum
            representable value for the specified `dtype`. Default is None.
        dtype (torch.dtype): Data type of the output mask. Default is `torch.float32`.
        cache_position (Optional[Int[torch.Tensor, "seq_len"]]): Absolute positions of the target tokens when
            using KV-cache during decoding. If None, positions are inferred from `past_key_values_length`.
            Note that a 1D cache_position with shape (1, seq_len), or a 2D cache_position with shape (#B, seq_len)
            are both acceptable. Default is None.
        past_key_values_length (int): Number of cached tokens already seen by the decoder. Default is 0.
        kv_seq_len (Optional[int]): Optional static key length (e.g., when using a compileable cache). If provided,
            the attention mask will be expanded or trimmed to this length.

    Returns:
        Float[torch.Tensor, "B 1 tgt_len key_len"]: Attention mask tensor where masked positions
            are set to `mask_value` (default to `-inf`) and unmasked positions are set to 0. Here
            `key_len` equals `max(seq_len, past_key_values_length + tgt_len, kv_seq_len or 0)`.
    """
    device = attention_mask.device
    batch_size, seq_len = attention_mask.shape
    tgt_len = tgt_len if tgt_len is not None else seq_len

    required_key_len = past_key_values_length + tgt_len
    if kv_seq_len is not None and kv_seq_len > 0:
        required_key_len = max(required_key_len, kv_seq_len)
    else:
        required_key_len = max(required_key_len, seq_len)

    if seq_len < required_key_len:
        pad = attention_mask.new_zeros(batch_size, required_key_len - seq_len)
        attention_mask = torch.cat([attention_mask, pad], dim=-1)
    elif seq_len > required_key_len:
        attention_mask = attention_mask[:, :required_key_len]

    padding_mask = attention_mask == 0
    padding_mask = padding_mask[:, None, None, :].expand(batch_size, 1, tgt_len, required_key_len)

    combined_mask = padding_mask
    if is_causal:
        if cache_position is None:
            cache_position = torch.arange(
                past_key_values_length,
                past_key_values_length + tgt_len,
                device=device,
            )
        cache_position = cache_position.to(device=device, dtype=torch.long)
        if cache_position.dim() == 1:  # pragma: no cover - defensive guard
            cache_position = cache_position.unsqueeze(0)
        if cache_position.dim() != 2:  # pragma: no cover - defensive guard
            raise ValueError("`cache_position` must be 1D or 2D tensor")
        if cache_position.size(-1) != tgt_len:  # pragma: no cover - defensive guard
            raise ValueError("`cache_position` length must match target length")
        if cache_position.size(0) == 1 and batch_size > 1:  # pragma: no cover - defensive guard
            cache_position = cache_position.expand(batch_size, -1)
        elif cache_position.size(0) not in {1, batch_size}:  # pragma: no cover - defensive guard
            raise ValueError("`cache_position` batch dimension must be 1 or match batch size")
        if cache_position.size(0) != batch_size:  # pragma: no cover - defensive guard
            cache_position = cache_position.expand(batch_size, -1)

        key_positions = torch.arange(required_key_len, device=device)
        causal_mask = cache_position.unsqueeze(-1) >= key_positions  # (B, tgt_len, key_len)
        causal_mask = causal_mask[:, None, :, :]
        combined_mask = combined_mask | ~causal_mask

    float_mask = torch.zeros_like(combined_mask, dtype=dtype, device=device)
    fill_value = mask_value if mask_value is not None else torch.finfo(dtype).min
    float_mask = float_mask.masked_fill(combined_mask, fill_value)

    return float_mask
