import pytest
import torch

from genrec.models.model_seqrec.hstu import HSTUModel, HSTUModelConfig


def test_hstu_forward_output_shapes() -> None:
    config = HSTUModelConfig(
        item_size=32,
        hidden_size=8,
        num_attention_heads=2,
        num_hidden_layers=2,
        enable_learnable_rel_posemb=False,
        enable_final_layernorm=True,
    )
    model = HSTUModel(config)

    batch_size, seq_len = 2, 5
    input_ids = torch.randint(1, config.item_size + 1, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    timestamps = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)

    output = model(input_ids=input_ids, attention_mask=attention_mask, timestamps=timestamps)

    assert output.last_hidden_state.shape == (batch_size, seq_len, config.hidden_size)
    assert output.model_loss is None
    assert output.hidden_states is None
    assert output.attentions is None


def test_hstu_returns_optional_collections_when_requested() -> None:
    config = HSTUModelConfig(
        item_size=24,
        hidden_size=12,
        num_attention_heads=3,
        num_hidden_layers=3,
        enable_input_pos_emb=True,
    )
    model = HSTUModel(config)

    batch_size, seq_len = 2, 4
    input_ids = torch.randint(1, config.item_size + 1, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    timestamps = torch.randint(0, 100, (batch_size, seq_len), dtype=torch.long)

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        timestamps=timestamps,
        output_hidden_states=True,
        output_attentions=True,
    )

    assert output.hidden_states is not None
    assert len(output.hidden_states) == config.num_hidden_layers + 1
    for layer_state in output.hidden_states:
        assert layer_state.shape == (batch_size, seq_len, config.hidden_size)

    assert output.attentions is not None
    assert len(output.attentions) == config.num_hidden_layers
    for attn in output.attentions:
        assert attn.shape == (batch_size, config.num_attention_heads, seq_len, seq_len)


def test_hstu_forward_requires_timestamps() -> None:
    config = HSTUModelConfig(
        item_size=16,
        hidden_size=8,
        num_attention_heads=2,
        num_hidden_layers=1,
    )
    model = HSTUModel(config)

    input_ids = torch.randint(1, config.item_size + 1, (1, 3))
    attention_mask = torch.ones_like(input_ids)

    with pytest.raises(ValueError):
        model(input_ids=input_ids, attention_mask=attention_mask)


def test_hstu_forward_without_input_positional_embeddings() -> None:
    config = HSTUModelConfig(
        item_size=20,
        hidden_size=10,
        num_attention_heads=2,
        num_hidden_layers=1,
        enable_input_pos_emb=False,
    )
    model = HSTUModel(config)

    assert model.input_pos_emb is None

    batch_size, seq_len = 2, 4
    input_ids = torch.randint(1, config.item_size + 1, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    timestamps = torch.randint(0, 50, (batch_size, seq_len), dtype=torch.long)

    output = model(input_ids=input_ids, attention_mask=attention_mask, timestamps=timestamps)

    assert output.last_hidden_state.shape == (batch_size, seq_len, config.hidden_size)
    assert output.hidden_states is None


def test_hstu_uses_gradient_checkpointing(monkeypatch) -> None:
    config = HSTUModelConfig(
        item_size=16,
        hidden_size=8,
        num_attention_heads=2,
        num_hidden_layers=2,
    )
    model = HSTUModel(config)
    model.gradient_checkpointing_enable()
    model.train()

    calls = []

    def fake_checkpoint(function, *args, **kwargs):
        calls.append(kwargs)
        return function(*args)

    monkeypatch.setattr("genrec.models.model_seqrec.hstu.checkpoint", fake_checkpoint)

    batch_size, seq_len = 1, 3
    input_ids = torch.randint(1, config.item_size + 1, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    timestamps = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)

    model(input_ids=input_ids, attention_mask=attention_mask, timestamps=timestamps)

    assert len(calls) == config.num_hidden_layers
    assert all(call.get("use_reentrant") is False for call in calls)
