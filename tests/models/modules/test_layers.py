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
