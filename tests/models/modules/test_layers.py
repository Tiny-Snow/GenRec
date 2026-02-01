import torch
from transformers.cache_utils import DynamicCache, EncoderDecoderCache

from genrec.models.modules.layers import LlamaDecoderLayer, SequentialTransductionUnit, T5Block
from genrec.models.modules.posemb import RotaryEmbedding
from genrec.models.modules.utils import create_attention_mask


def test_standard_decoder_layer_outputs_expected_shapes() -> None:
    hidden_size = 12
    num_heads = 3
    layer = LlamaDecoderLayer(
        hidden_size=hidden_size,
        num_heads=num_heads,
        intermediate_size=hidden_size * 2,
        attention_dropout=0.0,
        attention_bias=False,
        ffn_bias=False,
    )

    batch_size, seq_len = 2, 5
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    base_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    causal_mask = create_attention_mask(base_mask, is_causal=True)

    rotary = RotaryEmbedding(head_dim=layer.head_dim)
    position_embeddings = rotary(hidden_states)

    outputs, attn_weights = layer(
        hidden_states=hidden_states,
        attention_mask=causal_mask,
        position_embeddings=position_embeddings,
    )

    assert outputs.shape == (batch_size, seq_len, hidden_size)
    assert attn_weights.shape == (batch_size, num_heads, seq_len, seq_len)


def test_sequential_transduction_unit_outputs_expected_shapes() -> None:
    hidden_size = 16
    num_heads = 4
    seq_len = 6
    unit = SequentialTransductionUnit(
        hidden_size=hidden_size,
        num_heads=num_heads,
        intermediate_size=hidden_size * 2,
        attention_dropout=0.0,
        linear_dropout=0.0,
        max_seq_len=seq_len,
        num_buckets=8,
        enable_learnable_rel_posemb=True,
        enable_attention_gating=True,
        enable_ffn=False,
    )

    batch_size = 2
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    base_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    base_mask[:, -1] = 0
    attention_mask = create_attention_mask(base_mask, is_causal=True)
    timestamps = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)

    outputs, attn_weights = unit(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_embeddings=None,
        timestamps=timestamps,
    )

    assert outputs.shape == (batch_size, seq_len, hidden_size)
    assert attn_weights.shape == (batch_size, num_heads, seq_len, seq_len)
    assert torch.all(attn_weights[..., -1] == 0)


def test_sequential_transduction_unit_supports_rope_position_embeddings() -> None:
    hidden_size = 12
    num_heads = 3
    seq_len = 5
    unit = SequentialTransductionUnit(
        hidden_size=hidden_size,
        num_heads=num_heads,
        intermediate_size=hidden_size * 2,
        attention_dropout=0.0,
        linear_dropout=0.0,
        max_seq_len=seq_len,
        num_buckets=4,
        enable_learnable_rel_posemb=False,
        enable_attention_gating=True,
        enable_ffn=False,
    )

    batch_size = 2
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    rotary = RotaryEmbedding(head_dim=hidden_size // num_heads)
    position_embeddings = rotary(hidden_states)

    outputs, attn_weights = unit(
        hidden_states=hidden_states,
        attention_mask=None,
        position_embeddings=position_embeddings,
        timestamps=None,
    )

    assert outputs.shape == (batch_size, seq_len, hidden_size)
    assert attn_weights.shape == (batch_size, num_heads, seq_len, seq_len)


def test_sequential_transduction_unit_passes_timestamps_through_attention() -> None:
    hidden_size = 12
    num_heads = 3
    seq_len = 4
    unit = SequentialTransductionUnit(
        hidden_size=hidden_size,
        num_heads=num_heads,
        intermediate_size=hidden_size * 2,
        attention_dropout=0.0,
        linear_dropout=0.0,
        max_seq_len=seq_len,
        num_buckets=4,
        enable_learnable_rel_posemb=True,
        enable_attention_gating=True,
        enable_ffn=False,
    )

    class RecordingAttention(torch.nn.Module):
        def __init__(self, num_heads: int) -> None:
            super().__init__()
            self.num_heads = num_heads
            self.received_timestamps = None

        def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask=None,
            position_embeddings=None,
            timestamps=None,
        ):
            self.received_timestamps = timestamps
            batch, seq_len, _ = hidden_states.shape
            attn_weights = torch.zeros(
                batch,
                self.num_heads,
                seq_len,
                seq_len,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            return hidden_states, attn_weights

    recorder = RecordingAttention(num_heads=num_heads)
    unit.self_attn = recorder

    batch_size = 2
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    timestamps = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)

    outputs, attn_weights = unit(
        hidden_states=hidden_states,
        attention_mask=None,
        position_embeddings=None,
        timestamps=timestamps,
    )

    assert recorder.received_timestamps is not None
    assert torch.equal(recorder.received_timestamps, timestamps)
    assert outputs.shape == (batch_size, seq_len, hidden_size)
    assert attn_weights.shape == (batch_size, num_heads, seq_len, seq_len)


def test_sequential_transduction_unit_attention_gating_toggle_disables_gate() -> None:
    hidden_size = 10
    num_heads = 2
    seq_len = 4
    unit = SequentialTransductionUnit(
        hidden_size=hidden_size,
        num_heads=num_heads,
        intermediate_size=hidden_size * 2,
        attention_dropout=0.0,
        linear_dropout=0.0,
        max_seq_len=seq_len,
        num_buckets=4,
        enable_learnable_rel_posemb=True,
        enable_attention_gating=False,
        enable_ffn=False,
    )

    assert unit.enable_attention_gating is False
    assert unit.self_attn.enable_attention_gating is False
    assert unit.self_attn.u_proj is None
    assert unit.self_attn.av_output_layernorm is None

    batch_size = 1
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    timestamps = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

    outputs, _ = unit(
        hidden_states=hidden_states,
        attention_mask=None,
        position_embeddings=None,
        timestamps=timestamps,
    )

    assert outputs.shape == (batch_size, seq_len, hidden_size)


def test_sequential_transduction_unit_ffn_path_invokes_mlp() -> None:
    hidden_size = 14
    num_heads = 2
    seq_len = 4
    unit = SequentialTransductionUnit(
        hidden_size=hidden_size,
        num_heads=num_heads,
        intermediate_size=hidden_size * 2,
        attention_dropout=0.0,
        linear_dropout=0.0,
        max_seq_len=seq_len,
        num_buckets=4,
        enable_learnable_rel_posemb=True,
        enable_attention_gating=True,
        enable_ffn=True,
    )

    class RecordingMLP(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.called = False

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            self.called = True
            return hidden_states

    assert unit.mlp is not None
    unit.mlp = RecordingMLP()

    hidden_states = torch.randn(1, seq_len, hidden_size)
    timestamps = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

    _ = unit(
        hidden_states=hidden_states,
        attention_mask=None,
        position_embeddings=None,
        timestamps=timestamps,
    )

    assert isinstance(unit.mlp, RecordingMLP)
    assert unit.mlp.called


def test_t5_block_encoder_supports_masks_bias_and_rope() -> None:
    torch.manual_seed(0)

    hidden_size = 16
    num_heads = 4
    head_dim = hidden_size // num_heads
    block = T5Block(
        hidden_size=hidden_size,
        head_dim=head_dim,
        num_heads=num_heads,
        intermediate_size=hidden_size * 2,
        linear_dropout=0.0,
        attention_dropout=0.0,
        is_decoder=False,
        has_relative_attention_bias=False,
        enable_rope=True,
    )

    batch_size, seq_len = 2, 5
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)

    padding_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    padding_mask[1, -2:] = 0
    full_mask = create_attention_mask(padding_mask, is_causal=False)
    attention_mask = full_mask[:, :, :1, :]

    position_bias = torch.randn(1, num_heads, seq_len, seq_len)
    rotary = RotaryEmbedding(head_dim=head_dim)
    rope_embeddings = rotary(hidden_states)

    encoded_rope, self_outputs, cross_outputs = block(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_bias=position_bias,
        position_embeddings=rope_embeddings,
        output_attentions=True,
    )

    assert cross_outputs == (None, None)
    assert encoded_rope.shape == (batch_size, seq_len, hidden_size)

    self_position_bias, self_attn_weights = self_outputs
    assert self_position_bias is not None
    assert self_attn_weights is not None
    assert self_position_bias.shape == (batch_size, num_heads, seq_len, seq_len)
    assert self_attn_weights.shape == (batch_size, num_heads, seq_len, seq_len)

    mask_threshold = torch.finfo(self_position_bias.dtype).min / 2
    assert torch.all(self_position_bias[1, :, :, -2:] <= mask_threshold)
    assert torch.allclose(self_attn_weights[1, :, :, -2:], torch.zeros_like(self_attn_weights[1, :, :, -2:]))

    encoded_no_rope, _, _ = block(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_bias=position_bias.clone(),
        position_embeddings=None,
        output_attentions=False,
    )
    assert not torch.allclose(encoded_rope, encoded_no_rope)


def test_t5_block_decoder_cross_attention_and_cache_behaves_autoregressively() -> None:
    torch.manual_seed(1)

    hidden_size = 16
    num_heads = 4
    head_dim = hidden_size // num_heads
    prefill_len = 3
    encoder_len = 4

    block = T5Block(
        hidden_size=hidden_size,
        head_dim=head_dim,
        num_heads=num_heads,
        intermediate_size=hidden_size * 2,
        linear_dropout=0.0,
        attention_dropout=0.0,
        is_decoder=True,
        has_relative_attention_bias=False,
        enable_rope=True,
        layer_idx=0,
    )

    rotary = RotaryEmbedding(head_dim=head_dim)

    encoder_hidden_states = torch.randn(1, encoder_len, hidden_size)
    encoder_token_mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.long)
    encoder_attention_mask_full = create_attention_mask(encoder_token_mask, is_causal=False)
    encoder_attention_mask = encoder_attention_mask_full[:, :, :1, :]

    encoder_decoder_position_bias = torch.randn(1, num_heads, prefill_len, encoder_len)

    past_key_values = EncoderDecoderCache(DynamicCache(), DynamicCache())

    decoder_states = torch.randn(1, prefill_len, hidden_size)
    decoder_attention_mask = create_attention_mask(torch.ones(1, prefill_len, dtype=torch.long), is_causal=True)
    cache_position = torch.arange(prefill_len, dtype=torch.long)

    decoder_outputs = block(
        hidden_states=decoder_states,
        attention_mask=decoder_attention_mask,
        position_embeddings=rotary(decoder_states),
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask,
        encoder_decoder_position_bias=encoder_decoder_position_bias,
        past_key_values=past_key_values,
        cache_position=cache_position,
        output_attentions=True,
    )

    decoded_prefill, self_prefill, cross_prefill = decoder_outputs
    assert decoded_prefill.shape == (1, prefill_len, hidden_size)

    self_bias_prefill, self_attn_prefill = self_prefill
    assert self_bias_prefill is not None
    assert self_attn_prefill is not None
    assert self_bias_prefill.shape == (1, num_heads, prefill_len, prefill_len)
    assert self_attn_prefill.shape == (1, num_heads, prefill_len, prefill_len)

    cross_bias_prefill, cross_attn_prefill = cross_prefill
    assert cross_bias_prefill is not None
    assert cross_attn_prefill is not None
    assert cross_bias_prefill.shape == (1, num_heads, prefill_len, encoder_len)
    assert cross_attn_prefill.shape == (1, num_heads, prefill_len, encoder_len)

    mask_threshold = torch.finfo(cross_bias_prefill.dtype).min / 2
    assert torch.all(cross_bias_prefill[..., -1] <= mask_threshold)

    self_cache = past_key_values.self_attention_cache
    cross_cache = past_key_values.cross_attention_cache
    assert self_cache.layers[0].keys.shape == (1, num_heads, prefill_len, head_dim)
    assert cross_cache.layers[0].keys.shape == (1, num_heads, encoder_len, head_dim)
    cross_keys_before = cross_cache.layers[0].keys
    assert past_key_values.is_updated[0] is True

    next_state = torch.randn(1, 1, hidden_size)
    next_attention_mask = create_attention_mask(
        torch.ones(1, prefill_len + 1, dtype=torch.long),
        tgt_len=1,
        is_causal=True,
    )
    next_bias = torch.randn(1, num_heads, 1, encoder_len)
    next_cache_position = torch.tensor([prefill_len], dtype=torch.long)

    decoder_next_outputs = block(
        hidden_states=next_state,
        attention_mask=next_attention_mask,
        position_embeddings=rotary(next_state),
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask,
        encoder_decoder_position_bias=next_bias,
        past_key_values=past_key_values,
        cache_position=next_cache_position,
        output_attentions=True,
    )

    _, self_next, cross_next = decoder_next_outputs
    self_bias_next, self_attn_next = self_next
    cross_bias_next, cross_attn_next = cross_next

    assert self_bias_next is not None
    assert self_attn_next is not None
    assert cross_bias_next is not None
    assert cross_attn_next is not None

    assert self_bias_next.shape == (1, num_heads, 1, prefill_len + 1)
    assert cross_bias_next.shape == (1, num_heads, 1, encoder_len)
    assert self_attn_next.shape == (1, num_heads, 1, prefill_len + 1)
    assert cross_attn_next.shape == (1, num_heads, 1, encoder_len)

    assert self_cache.layers[0].keys.shape[-2] == prefill_len + 1
    assert cross_cache.layers[0].keys is cross_keys_before
