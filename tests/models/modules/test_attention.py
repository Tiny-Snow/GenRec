from __future__ import annotations

import torch

from genrec.models.modules.attention import MaskedSelfAttentionWithRoPE, T5Attention
from genrec.models.modules.posemb import RotaryEmbedding
from genrec.models.modules.utils import create_attention_mask


class _MockCacheLayer:
    """Lightweight container to mimic cached key/value tensors."""

    def __init__(self) -> None:
        self.keys: torch.Tensor | None = None
        self.values: torch.Tensor | None = None


class _MockCache:
    """Minimal cache that supports the DynamicCache interface subset."""

    def __init__(self, num_layers: int) -> None:
        self.layers = [_MockCacheLayer() for _ in range(num_layers)]

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, info: dict | None):
        cache_position = info.get("cache_position") if info is not None else None
        layer = self.layers[layer_idx]
        if layer.keys is None:
            layer.keys = key_states
            layer.values = value_states
        else:
            should_append = cache_position is None
            if cache_position is not None:
                should_append = int(cache_position.min().item()) >= layer.keys.size(-2)
            if should_append:
                layer.keys = torch.cat([layer.keys, key_states], dim=-2)
                layer.values = torch.cat([layer.values, value_states], dim=-2)
        return layer.keys, layer.values


class _MockEncoderDecoderCache:
    def __init__(self, num_layers: int) -> None:
        self.is_updated = {idx: False for idx in range(num_layers)}
        self.self_attention_cache = _MockCache(num_layers)
        self.cross_attention_cache = _MockCache(num_layers)


def test_masked_self_attention_with_rope_preserves_shapes() -> None:
    hidden_size = 8
    num_heads = 2
    head_dim = hidden_size // num_heads

    attention = MaskedSelfAttentionWithRoPE(
        hidden_size=hidden_size,
        head_dim=head_dim,
        num_heads=num_heads,
        attention_dropout=0.0,
        attention_bias=False,
    )

    batch_size, seq_len = 2, 4
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    base_mask = torch.ones(batch_size, seq_len, dtype=torch.float32)
    causal_mask = create_attention_mask(base_mask, is_causal=True)

    rotary = RotaryEmbedding(head_dim=head_dim)
    position_embeddings = rotary(torch.zeros(batch_size, seq_len, head_dim))

    attn_output, attn_weights = attention(
        hidden_states,
        attention_mask=causal_mask,
        position_embeddings=position_embeddings,
    )

    assert attn_output.shape == (batch_size, seq_len, hidden_size)
    assert attn_weights.shape == (batch_size, num_heads, seq_len, seq_len)


def test_masked_self_attention_without_rope_or_mask_behaves_well() -> None:
    hidden_size = 6
    num_heads = 3
    head_dim = hidden_size // num_heads

    attention = MaskedSelfAttentionWithRoPE(
        hidden_size=hidden_size,
        head_dim=head_dim,
        num_heads=num_heads,
        attention_dropout=0.0,
        attention_bias=True,
    )
    attention.eval()  # deterministic dropout branch

    batch_size, seq_len = 1, 3
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)

    attn_output, attn_weights = attention(hidden_states)

    assert attn_output.shape == (batch_size, seq_len, hidden_size)
    assert attn_weights.shape == (batch_size, num_heads, seq_len, seq_len)
    row_sums = attn_weights.sum(dim=-1)
    torch.testing.assert_close(row_sums, torch.ones_like(row_sums))


def test_t5_attention_with_relative_bias_and_padding_mask() -> None:
    hidden_size = 12
    num_heads = 3
    head_dim = hidden_size // num_heads

    attention = T5Attention(
        hidden_size=hidden_size,
        head_dim=head_dim,
        num_heads=num_heads,
        attention_dropout=0.0,
        attention_bias=True,
        is_decoder=False,
        has_relative_attention_bias=True,
        enable_rope=False,
    )
    attention.eval()

    batch_size, seq_len = 2, 5
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    base_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]], dtype=torch.long)
    attention_mask = create_attention_mask(base_mask, is_causal=False)

    attn_output, position_bias, attn_weights = attention(
        hidden_states,
        attention_mask=attention_mask,
        output_attentions=True,
    )

    assert attn_output.shape == (batch_size, seq_len, hidden_size)
    assert attn_weights.shape == (batch_size, num_heads, seq_len, seq_len)
    assert position_bias is not None
    assert torch.count_nonzero(position_bias).item() > 0
    torch.testing.assert_close(attn_weights[..., -1], torch.zeros_like(attn_weights[..., -1]))


def test_t5_attention_switches_to_rope_when_bias_disabled() -> None:
    torch.manual_seed(42)
    hidden_size = 12
    num_heads = 2
    head_dim = hidden_size // num_heads

    rotary = RotaryEmbedding(head_dim=head_dim)

    torch.manual_seed(123)
    attention_with_rope = T5Attention(
        hidden_size=hidden_size,
        head_dim=head_dim,
        num_heads=num_heads,
        attention_dropout=0.0,
        attention_bias=False,
        has_relative_attention_bias=False,
        enable_rope=True,
    )
    attention_with_rope.eval()

    torch.manual_seed(123)
    attention_without_rope = T5Attention(
        hidden_size=hidden_size,
        head_dim=head_dim,
        num_heads=num_heads,
        attention_dropout=0.0,
        attention_bias=False,
        has_relative_attention_bias=False,
        enable_rope=False,
    )
    attention_without_rope.eval()

    batch_size, seq_len = 1, 4
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    position_embeddings = rotary(torch.zeros(batch_size, seq_len, head_dim))

    rope_output, rope_bias, _ = attention_with_rope(hidden_states, position_embeddings=position_embeddings)
    plain_output, plain_bias, _ = attention_without_rope(hidden_states)

    assert rope_bias is not None and torch.count_nonzero(rope_bias).item() == 0
    assert plain_bias is not None and torch.count_nonzero(plain_bias).item() == 0
    assert rope_output.shape == plain_output.shape
    assert not torch.allclose(rope_output, plain_output)


def test_t5_attention_cross_attention_cache_reuse() -> None:
    hidden_size = 9
    num_heads = 3
    head_dim = hidden_size // num_heads

    attention = T5Attention(
        hidden_size=hidden_size,
        head_dim=head_dim,
        num_heads=num_heads,
        attention_dropout=0.0,
        attention_bias=False,
        is_decoder=True,
        layer_idx=0,
    )
    attention.eval()

    cache = _MockEncoderDecoderCache(num_layers=1)

    batch_size, query_len, key_len = 1, 2, 4
    hidden_states = torch.randn(batch_size, query_len, hidden_size)
    encoder_states = torch.randn(batch_size, key_len, hidden_size)
    overlong_mask = torch.ones(batch_size, key_len + 2, dtype=torch.long)
    attention_mask = create_attention_mask(overlong_mask, tgt_len=query_len, is_causal=False)

    attn_output, position_bias, attn_weights = attention(
        hidden_states,
        attention_mask=attention_mask,
        key_value_states=encoder_states,
        past_key_values=cache,
        output_attentions=True,
    )

    assert attn_output.shape == (batch_size, query_len, hidden_size)
    assert attn_weights.shape[-1] == key_len
    assert position_bias is not None and position_bias.shape[-1] == key_len
    assert cache.is_updated[0]

    cached_keys = cache.cross_attention_cache.layers[0].keys
    cached_values = cache.cross_attention_cache.layers[0].values
    assert cached_keys is not None and cached_keys.shape[-2] == key_len
    assert cached_values is not None and cached_values.shape[-2] == key_len

    zero_encoder = torch.zeros_like(encoder_states)
    cached_output, _, _ = attention(
        hidden_states,
        attention_mask=attention_mask,
        key_value_states=zero_encoder,
        past_key_values=cache,
        output_attentions=True,
    )

    torch.testing.assert_close(attn_output, cached_output)


def test_t5_attention_self_attention_cache_position_appends_tokens() -> None:
    hidden_size = 8
    num_heads = 2
    head_dim = hidden_size // num_heads

    attention = T5Attention(
        hidden_size=hidden_size,
        head_dim=head_dim,
        num_heads=num_heads,
        attention_dropout=0.0,
        attention_bias=False,
        is_decoder=True,
        has_relative_attention_bias=True,
        enable_rope=False,
        layer_idx=0,
    )
    attention.eval()

    cache = _MockEncoderDecoderCache(num_layers=1)

    first_hidden = torch.randn(1, 2, hidden_size)
    first_mask = create_attention_mask(torch.ones(1, 2, dtype=torch.long), is_causal=True)
    first_positions = torch.arange(0, 2)

    _, first_bias, first_weights = attention(
        first_hidden,
        attention_mask=first_mask,
        past_key_values=cache,
        cache_position=first_positions,
        output_attentions=True,
    )

    assert first_weights.shape[-1] == 2
    assert first_bias is not None and first_bias.shape[-1] == 2

    second_hidden = torch.randn(1, 1, hidden_size)
    # key length becomes 3 after appending cached tokens
    decoder_memory_mask = torch.ones(1, 3, dtype=torch.long)
    second_mask = create_attention_mask(decoder_memory_mask, tgt_len=1, is_causal=True)
    second_positions = torch.tensor([2])

    _, second_bias, second_weights = attention(
        second_hidden,
        attention_mask=second_mask,
        past_key_values=cache,
        cache_position=second_positions,
        output_attentions=True,
    )

    cached_layer = cache.self_attention_cache.layers[0]
    assert cached_layer.keys is not None and cached_layer.keys.shape[-2] == 3
    assert cached_layer.values is not None and cached_layer.values.shape[-2] == 3
    assert second_weights.shape[-1] == 3
    assert second_bias is not None and second_bias.shape[-1] == 3


def test_t5_attention_zero_bias_branch_and_mask_slicing() -> None:
    hidden_size = 8
    num_heads = 2
    head_dim = hidden_size // num_heads

    attention = T5Attention(
        hidden_size=hidden_size,
        head_dim=head_dim,
        num_heads=num_heads,
        attention_dropout=0.0,
        attention_bias=False,
        has_relative_attention_bias=False,
    )
    attention.eval()

    batch_size, seq_len = 1, 3
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)

    # create a mask whose key dimension is longer than the actual key length to trigger slicing
    base_mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.long)
    attention_mask = create_attention_mask(base_mask, tgt_len=seq_len, is_causal=False)

    attn_output, position_bias, attn_weights = attention(
        hidden_states,
        attention_mask=attention_mask,
        output_attentions=True,
    )

    assert attn_output.shape == (batch_size, seq_len, hidden_size)
    assert attn_weights.shape == (batch_size, num_heads, seq_len, seq_len)
    assert position_bias is not None and position_bias.shape[-1] == seq_len

    expected_mask = attention_mask[:, :, :, :seq_len]
    expanded_mask = expected_mask.expand(-1, num_heads, -1, -1)
    torch.testing.assert_close(position_bias, expanded_mask)


def test_t5_attention_relative_bias_branch_computes_and_adds_bias() -> None:
    torch.manual_seed(7)
    hidden_size = 16
    num_heads = 4
    head_dim = hidden_size // num_heads

    attention = T5Attention(
        hidden_size=hidden_size,
        head_dim=head_dim,
        num_heads=num_heads,
        attention_dropout=0.0,
        attention_bias=True,
        has_relative_attention_bias=True,
        enable_rope=False,
    )
    attention.eval()

    batch_size, seq_len = 1, 5
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)

    attn_output_with_bias, position_bias, attn_weights_with_bias = attention(
        hidden_states,
        output_attentions=True,
    )

    assert attn_output_with_bias.shape == (batch_size, seq_len, hidden_size)
    assert attn_weights_with_bias.shape == (batch_size, num_heads, seq_len, seq_len)
    assert position_bias is not None

    expected_bias = attention.rel_pos_bias(seq_len, seq_len, device=hidden_states.device)
    expected_bias = expected_bias[:, :, -seq_len:, :]
    torch.testing.assert_close(position_bias, expected_bias)

    zero_bias = torch.zeros_like(position_bias)
    _, _, attn_weights_without_bias = attention(
        hidden_states,
        position_bias=zero_bias,
        output_attentions=True,
    )

    assert not torch.allclose(attn_weights_with_bias, attn_weights_without_bias)
