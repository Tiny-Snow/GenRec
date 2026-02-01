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


def test_create_attention_mask_respects_cache_positions_and_past() -> None:
    attention_mask = torch.ones(1, 6, dtype=torch.float32)
    cache_position = torch.tensor([[3, 4]])

    mask = create_attention_mask(
        attention_mask,
        tgt_len=2,
        is_causal=True,
        cache_position=cache_position,
        past_key_values_length=3,
    )

    assert mask.shape == (1, 1, 2, 6)
    min_value = torch.finfo(mask.dtype).min
    assert mask[0, 0, 0, 4] == min_value  # cannot attend to future key
    assert mask[0, 0, 0, 2] == 0  # can attend to earlier key
    assert mask[0, 0, 1, 4] == 0  # later query can attend to aligned key


def test_create_attention_mask_extends_with_kv_seq_len() -> None:
    attention_mask = torch.ones(1, 3, dtype=torch.float32)

    mask = create_attention_mask(
        attention_mask,
        tgt_len=1,
        is_causal=True,
        kv_seq_len=6,
    )

    assert mask.shape == (1, 1, 1, 6)
    min_value = torch.finfo(mask.dtype).min
    assert mask[0, 0, 0, 0] == 0
    assert mask[0, 0, 0, 5] == min_value  # padded keys beyond seq_len are masked


def test_create_attention_mask_trims_when_kv_seq_len_is_smaller() -> None:
    attention_mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.float32)

    mask = create_attention_mask(
        attention_mask,
        tgt_len=1,
        is_causal=False,
        kv_seq_len=3,
    )

    assert mask.shape == (1, 1, 1, 3)
    assert torch.all(mask == 0)
