import pytest
import torch

from genrec.models.modules.layers import LlamaDecoderLayer, SequentialTransductionUnit
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
    base_mask = torch.ones(batch_size, seq_len, dtype=torch.float32)
    causal_mask = create_attention_mask(base_mask, is_causal=True)

    rotary = RotaryEmbedding(head_dim=layer.head_dim)
    position_embeddings = rotary(torch.zeros(batch_size, seq_len, layer.head_dim))

    outputs, attn_weights = layer(
        hidden_states=hidden_states,
        attention_mask=causal_mask,
        position_embeddings=position_embeddings,
    )

    assert outputs.shape == (batch_size, seq_len, hidden_size)
    assert attn_weights.shape == (batch_size, num_heads, seq_len, seq_len)


def test_sequential_transduction_unit_respects_attention_mask() -> None:
    hidden_size = 8
    num_heads = 2
    seq_len = 4
    unit = SequentialTransductionUnit(
        hidden_size=hidden_size,
        num_heads=num_heads,
        max_seq_len=seq_len,
        num_buckets=4,
        linear_dropout=0.0,
        attention_dropout=0.0,
    )

    batch_size = 2
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    attention_mask = torch.ones(batch_size, 1, seq_len, seq_len, dtype=torch.bool)
    attention_mask[:, :, :, -1] = False
    timestamps = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)

    outputs, attn_weights = unit(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        timestamps=timestamps,
    )

    assert outputs.shape == (batch_size, seq_len, hidden_size)
    assert attn_weights.shape == (batch_size, num_heads, seq_len, seq_len)
    assert torch.all(attn_weights[..., -1] == 0)


def test_sequential_transduction_unit_handles_missing_timestamps_and_mask() -> None:
    hidden_size = 6
    num_heads = 2
    seq_len = 3
    unit = SequentialTransductionUnit(
        hidden_size=hidden_size,
        num_heads=num_heads,
        max_seq_len=seq_len,
        num_buckets=2,
        linear_dropout=0.0,
        attention_dropout=0.0,
    )

    batch_size = 1
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)

    outputs, attn_weights = unit(
        hidden_states=hidden_states,
        attention_mask=None,
        timestamps=None,
    )

    assert outputs.shape == (batch_size, seq_len, hidden_size)
    assert attn_weights.shape == (batch_size, num_heads, seq_len, seq_len)


def test_sequential_transduction_unit_softmax_attention_normalizes_weights() -> None:
    hidden_size = 8
    num_heads = 2
    seq_len = 3
    unit = SequentialTransductionUnit(
        hidden_size=hidden_size,
        num_heads=num_heads,
        max_seq_len=seq_len,
        num_buckets=4,
        linear_dropout=0.0,
        attention_dropout=0.0,
        softmax_attention=True,
    )
    unit.eval()

    batch_size = 2
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)

    _, attn_weights = unit(
        hidden_states=hidden_states,
        attention_mask=None,
        timestamps=None,
    )

    assert attn_weights.shape == (batch_size, num_heads, seq_len, seq_len)
    row_sums = attn_weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)
    assert torch.all(attn_weights >= 0)


def test_sequential_transduction_unit_attention_norm_normalizes_weights() -> None:
    hidden_size = 8
    num_heads = 2
    seq_len = 4
    unit = SequentialTransductionUnit(
        hidden_size=hidden_size,
        num_heads=num_heads,
        max_seq_len=seq_len,
        num_buckets=4,
        linear_dropout=0.0,
        attention_dropout=0.0,
        softmax_attention=False,
        attention_norm=True,
    )
    unit.eval()

    batch_size = 3
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)

    _, attn_weights = unit(
        hidden_states=hidden_states,
        attention_mask=None,
        timestamps=None,
    )

    assert attn_weights.shape == (batch_size, num_heads, seq_len, seq_len)
    row_sums = attn_weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_sequential_transduction_unit_time_interval_matches_manual_scaling() -> None:
    hidden_size = 6
    num_heads = 2
    seq_len = 4
    time_interval = 86_400

    unit_seconds = SequentialTransductionUnit(
        hidden_size=hidden_size,
        num_heads=num_heads,
        max_seq_len=seq_len,
        num_buckets=8,
        linear_dropout=0.0,
        attention_dropout=0.0,
        time_interval=time_interval,
    )
    unit_scaled = SequentialTransductionUnit(
        hidden_size=hidden_size,
        num_heads=num_heads,
        max_seq_len=seq_len,
        num_buckets=8,
        linear_dropout=0.0,
        attention_dropout=0.0,
        time_interval=1.0,
    )
    unit_scaled.load_state_dict(unit_seconds.state_dict())
    unit_seconds.eval()
    unit_scaled.eval()

    batch_size = 2
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    timestamps_seconds = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1) * time_interval
    timestamps_scaled = timestamps_seconds // time_interval

    out_seconds, attn_seconds = unit_seconds(
        hidden_states=hidden_states,
        attention_mask=None,
        timestamps=timestamps_seconds,
    )
    out_scaled, attn_scaled = unit_scaled(
        hidden_states=hidden_states,
        attention_mask=None,
        timestamps=timestamps_scaled,
    )

    assert torch.allclose(out_seconds, out_scaled, atol=1e-5)
    assert torch.allclose(attn_seconds, attn_scaled, atol=1e-5)


def test_sequential_transduction_unit_rejects_non_positive_time_interval() -> None:
    with pytest.raises(ValueError):
        SequentialTransductionUnit(
            hidden_size=4,
            num_heads=2,
            max_seq_len=4,
            num_buckets=2,
            linear_dropout=0.0,
            attention_dropout=0.0,
            time_interval=0.0,
        )
