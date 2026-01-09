import pytest
import torch

from genrec.models.model_seqrec.hstu import HSTUModel, HSTUModelConfig


def _dummy_inputs(
    config: HSTUModelConfig, batch_size: int = 2, seq_len: int = 5
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids = torch.randint(1, config.item_size + 1, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    timestamps = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    return input_ids, attention_mask, timestamps


def test_hstu_forward_output_shapes() -> None:
    config = HSTUModelConfig(
        item_size=32,
        hidden_size=12,
        num_attention_heads=3,
        num_hidden_layers=2,
        max_seq_len=16,
    )
    model = HSTUModel(config)

    input_ids, attention_mask, timestamps = _dummy_inputs(config)
    output = model(input_ids=input_ids, attention_mask=attention_mask, timestamps=timestamps)

    batch_size, seq_len = input_ids.shape
    assert output.last_hidden_state.shape == (batch_size, seq_len, config.hidden_size)
    assert output.model_loss is None
    assert output.hidden_states is None
    assert output.attentions is None


def test_hstu_requires_timestamps_argument() -> None:
    config = HSTUModelConfig(item_size=8, hidden_size=6, num_attention_heads=2, num_hidden_layers=1)
    model = HSTUModel(config)

    input_ids, attention_mask, _ = _dummy_inputs(config, batch_size=1, seq_len=3)

    with pytest.raises(ValueError):
        model(input_ids=input_ids, attention_mask=attention_mask)


def test_hstu_returns_hidden_states_and_attentions() -> None:
    config = HSTUModelConfig(
        item_size=24,
        hidden_size=8,
        num_attention_heads=2,
        num_hidden_layers=2,
        max_seq_len=12,
        final_layer_norm=True,
    )
    model = HSTUModel(config)

    input_ids, attention_mask, timestamps = _dummy_inputs(config, batch_size=2, seq_len=4)
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        timestamps=timestamps,
        output_hidden_states=True,
        output_attentions=True,
    )

    assert output.hidden_states is not None
    assert len(output.hidden_states) == config.num_hidden_layers + 1
    for hidden_state in output.hidden_states:
        assert hidden_state.shape == (input_ids.size(0), input_ids.size(1), config.hidden_size)

    assert output.attentions is not None
    assert len(output.attentions) == config.num_hidden_layers
    for attn in output.attentions:
        assert attn.shape == (
            input_ids.size(0),
            config.num_attention_heads,
            input_ids.size(1),
            input_ids.size(1),
        )


def test_hstu_runs_optional_ffn_path() -> None:
    config = HSTUModelConfig(
        item_size=16,
        hidden_size=8,
        num_attention_heads=2,
        num_hidden_layers=1,
        max_seq_len=8,
        add_ffn=True,
    )
    model = HSTUModel(config)

    assert model.layers[0].mlp is not None

    class RecordingMLP(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.called = False

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            self.called = True
            return torch.zeros_like(hidden_states)

    recording_mlp = RecordingMLP()
    model.layers[0].mlp = recording_mlp

    input_ids, attention_mask, timestamps = _dummy_inputs(config, batch_size=1, seq_len=4)
    output = model(input_ids=input_ids, attention_mask=attention_mask, timestamps=timestamps)

    assert recording_mlp.called
    assert output.last_hidden_state.shape == (1, 4, config.hidden_size)


def test_hstu_attention_gating_toggle_propagates() -> None:
    config = HSTUModelConfig(
        item_size=10,
        hidden_size=6,
        num_attention_heads=2,
        num_hidden_layers=1,
        attention_gating=False,
    )
    model = HSTUModel(config)

    assert all(layer.attention_gating is False for layer in model.layers)
    assert all(layer.u_proj is None for layer in model.layers)
    assert all(layer.attn_output_layernorm is None for layer in model.layers)

    input_ids, attention_mask, timestamps = _dummy_inputs(config, batch_size=1, seq_len=3)
    output = model(input_ids=input_ids, attention_mask=attention_mask, timestamps=timestamps)

    assert output.last_hidden_state.shape == (1, 3, config.hidden_size)
