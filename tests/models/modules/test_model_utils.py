import torch

from genrec.models.modules.utils import create_attention_mask


def test_create_attention_mask_handles_padding_tokens() -> None:
    attention_mask = torch.tensor([[1, 1, 0]], dtype=torch.float32)
    mask = create_attention_mask(attention_mask, is_causal=False)

    assert mask.shape == (1, 1, 3, 3)
    min_value = torch.finfo(mask.dtype).min
    assert torch.all(mask[0, 0, :, 2] == min_value)
    assert torch.all(mask[0, 0, :, :2] == 0)


def test_create_attention_mask_applies_causal_blocking_and_tgt_len() -> None:
    attention_mask = torch.ones(1, 4, dtype=torch.float32)
    mask = create_attention_mask(attention_mask, tgt_len=2, is_causal=True)

    assert mask.shape == (1, 1, 2, 4)
    min_value = torch.finfo(mask.dtype).min
    assert mask[0, 0, 0, 1] == min_value
    assert mask[0, 0, 1, 3] == min_value
    assert mask[0, 0, 1, 0] == 0
