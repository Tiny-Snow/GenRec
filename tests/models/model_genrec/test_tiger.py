from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from types import MethodType

from genrec.datasets import DatasetSplitLiteral, GenRecCollator, GenRecDataset
from genrec.models.model_genrec.tiger import TIGERModel, TIGERModelConfig


def _make_interaction_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "UserID": np.array([0, 1, 2], dtype=np.int64),
            "ItemID": [
                [1, 2, 3, 4, 5],
                [2, 3, 4, 5, 6],
                [3, 4, 5, 6, 7],
            ],
            "Timestamp": [
                [100, 101, 102, 103, 104],
                [200, 201, 202, 203, 204],
                [300, 301, 302, 303, 304],
            ],
        }
    )


def _make_sid_cache(num_items: int, sid_width: int = 3) -> np.ndarray:
    cache = np.zeros((num_items + 1, sid_width), dtype=np.int64)
    for item_id in range(1, num_items + 1):
        base = item_id * sid_width
        cache[item_id] = np.arange(base, base + sid_width, dtype=np.int64)
    return cache


def _build_batch(batch_size: int = 2) -> tuple[dict[str, torch.Tensor], np.ndarray]:
    sid_cache = _make_sid_cache(num_items=7, sid_width=3)
    dataset = GenRecDataset(
        interaction_data_path=_make_interaction_frame(),
        split=DatasetSplitLiteral.TRAIN,
        max_seq_length=3,
        min_seq_length=1,
        sid_cache=sid_cache,
    )
    collator = GenRecCollator(dataset)
    examples = [dataset[idx] for idx in range(batch_size)]
    batch = collator(examples)
    return batch, sid_cache


def _make_config(
    vocab_size: int,
    *,
    enable_rope: bool = False,
    tie_word_embeddings: bool = True,
) -> TIGERModelConfig:
    return TIGERModelConfig(
        hidden_size=32,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        linear_dropout=0.1,
        attention_dropout=0.0,
        attention_bias=False,
        ffn_bias=False,
        enable_rope=enable_rope,
        vocab_size=vocab_size,
        decoder_start_token_id=0,
        pad_token_id=0,
        dropout_rate=0.0,
        use_cache=True,
        tie_word_embeddings=tie_word_embeddings,
    )


def test_tiger_model_forward_matches_batch_shapes() -> None:
    batch, sid_cache = _build_batch(batch_size=2)
    vocab_size = int(np.max(sid_cache) + 32)
    model = TIGERModel(_make_config(vocab_size))

    output = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
        decoder_attention_mask=torch.ones_like(batch["labels"]),
    )

    batch_size, enc_seq_len = batch["input_ids"].shape
    label_width = batch["labels"].shape[1]

    assert output.logits.shape == (batch_size, label_width, vocab_size)
    assert output.encoder_last_hidden_state.shape == (batch_size, enc_seq_len, model.config.hidden_size)
    assert output.past_key_values is not None
    assert output.model_loss is None


def test_tiger_model_returns_optional_collections() -> None:
    batch, sid_cache = _build_batch(batch_size=2)
    vocab_size = int(np.max(sid_cache) + 32)
    model = TIGERModel(_make_config(vocab_size))

    output = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
        decoder_attention_mask=torch.ones_like(batch["labels"]),
        output_hidden_states=True,
        output_attentions=True,
    )

    assert output.decoder_hidden_states is not None
    assert len(output.decoder_hidden_states) == model.config.num_decoder_layers + 1
    for layer_state in output.decoder_hidden_states:
        assert layer_state.shape[-1] == model.config.hidden_size

    assert output.decoder_attentions is not None
    assert len(output.decoder_attentions) == model.config.num_decoder_layers
    for attn in output.decoder_attentions:
        assert attn.shape[1] == model.config.num_heads

    assert output.cross_attentions is not None
    assert len(output.cross_attentions) == model.config.num_decoder_layers

    assert output.encoder_hidden_states is not None
    assert len(output.encoder_hidden_states) == model.config.num_encoder_layers + 1

    assert output.encoder_attentions is not None
    assert len(output.encoder_attentions) == model.config.num_encoder_layers


def test_tiger_model_reuses_past_key_values_for_incremental_generation() -> None:
    batch, sid_cache = _build_batch(batch_size=1)
    vocab_size = int(np.max(sid_cache) + 32)
    model = TIGERModel(_make_config(vocab_size))

    first_step = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
        decoder_attention_mask=torch.ones_like(batch["labels"]),
        use_cache=True,
    )

    assert first_step.past_key_values is not None
    initial_decoder_length = first_step.past_key_values.get_seq_length()
    assert initial_decoder_length == batch["labels"].shape[1]

    next_token = batch["labels"][:, -1:]
    second_step = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        decoder_input_ids=next_token,
        decoder_attention_mask=torch.ones_like(next_token),
        past_key_values=first_step.past_key_values,
        use_cache=True,
    )

    assert second_step.logits.shape == (1, 1, vocab_size)
    assert second_step.past_key_values is not None
    assert second_step.past_key_values.get_seq_length() == initial_decoder_length + 1


def test_tiger_model_uses_provided_encoder_outputs() -> None:
    batch, sid_cache = _build_batch(batch_size=1)
    vocab_size = int(np.max(sid_cache) + 32)
    model = TIGERModel(_make_config(vocab_size))

    with torch.no_grad():
        encoder_outputs = model.encoder(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            output_hidden_states=True,
            output_attentions=True,
        )

    decoder_input_ids = batch["labels"]
    decoder_attention_mask = torch.ones_like(decoder_input_ids)
    output = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=decoder_attention_mask,
        encoder_outputs=encoder_outputs,
    )

    assert output.encoder_last_hidden_state is encoder_outputs.last_hidden_state
    assert output.encoder_attentions is encoder_outputs.attentions
    assert output.encoder_hidden_states is encoder_outputs.hidden_states


def test_tiger_model_returns_model_loss_when_requested() -> None:
    batch, sid_cache = _build_batch(batch_size=1)
    vocab_size = int(np.max(sid_cache) + 32)
    model = TIGERModel(_make_config(vocab_size))

    output = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
        decoder_attention_mask=torch.ones_like(batch["labels"]),
        output_model_loss=True,
    )

    assert output.model_loss is not None
    torch.testing.assert_close(output.model_loss, torch.tensor(0.0, device=output.model_loss.device))


def test_tiger_encoder_stack_rejects_cache_usage() -> None:
    batch, sid_cache = _build_batch(batch_size=1)
    vocab_size = int(np.max(sid_cache) + 32)
    model = TIGERModel(_make_config(vocab_size))

    with pytest.raises(ValueError):
        model.encoder(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=True,
        )


def test_tiger_decoder_turns_off_cache_with_gradient_checkpointing() -> None:
    batch, sid_cache = _build_batch(batch_size=1)
    vocab_size = int(np.max(sid_cache) + 32)
    model = TIGERModel(_make_config(vocab_size))

    encoder_outputs = model.encoder(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )

    decoder = model.decoder
    decoder.gradient_checkpointing_enable()
    decoder.train()

    outputs = decoder(
        input_ids=batch["labels"],
        attention_mask=torch.ones_like(batch["labels"]),
        encoder_hidden_states=encoder_outputs.last_hidden_state,
        encoder_attention_mask=batch["attention_mask"],
        use_cache=True,
    )

    assert outputs.past_key_values is None


def test_tiger_decoder_builds_attention_mask_when_missing() -> None:
    batch, sid_cache = _build_batch(batch_size=1)
    vocab_size = int(np.max(sid_cache) + 32)
    model = TIGERModel(_make_config(vocab_size))

    encoder_outputs = model.encoder(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )

    outputs = model.decoder(
        input_ids=batch["labels"],
        attention_mask=None,
        encoder_hidden_states=encoder_outputs.last_hidden_state,
        encoder_attention_mask=batch["attention_mask"],
    )

    assert outputs.last_hidden_state.shape[0] == batch["labels"].shape[0]


def test_tiger_decoder_passes_cache_position_to_rope() -> None:
    batch, sid_cache = _build_batch(batch_size=2)
    vocab_size = int(np.max(sid_cache) + 32)
    model = TIGERModel(_make_config(vocab_size, enable_rope=True))

    class RecordingRotary(torch.nn.Module):
        def __init__(self, head_dim: int) -> None:
            super().__init__()
            self.position_ids: torch.Tensor | None = None
            self.head_dim = head_dim

        def forward(
            self,
            hidden_states: torch.Tensor,
            position_ids: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            if position_ids is not None:
                self.position_ids = position_ids.clone()
            sliced = hidden_states[..., : self.head_dim]
            return sliced, sliced

    recorder = RecordingRotary(model.config.head_dim)
    model.decoder.rotary_emb = recorder

    encoder_outputs = model.encoder(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )
    cache_position = torch.arange(batch["labels"].shape[1], device=batch["labels"].device)

    model.decoder(
        input_ids=batch["labels"],
        attention_mask=torch.ones_like(batch["labels"]),
        encoder_hidden_states=encoder_outputs.last_hidden_state,
        encoder_attention_mask=batch["attention_mask"],
        cache_position=cache_position,
    )

    assert recorder.position_ids is not None
    assert recorder.position_ids.shape == (batch["labels"].shape[0], batch["labels"].shape[1])
    expected = cache_position.unsqueeze(0).expand(batch["labels"].shape[0], -1)
    assert torch.equal(recorder.position_ids, expected)


def test_tiger_model_set_input_embeddings_propagates_to_stacks() -> None:
    vocab_size = 64
    model = TIGERModel(_make_config(vocab_size))

    new_embeddings = nn.Embedding(vocab_size, model.config.hidden_size)
    model.set_input_embeddings(new_embeddings)

    assert model.get_input_embeddings() is new_embeddings
    assert model.encoder.embed_tokens is new_embeddings
    assert model.decoder.embed_tokens is new_embeddings


def test_tiger_decoder_supports_inputs_embeds_path() -> None:
    batch, sid_cache = _build_batch(batch_size=1)
    vocab_size = int(np.max(sid_cache) + 32)
    model = TIGERModel(_make_config(vocab_size))

    encoder_outputs = model.encoder(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )

    decoder_embeds = model.decoder.embed_tokens(batch["labels"])
    outputs = model.decoder(
        input_ids=None,
        inputs_embeds=decoder_embeds,
        attention_mask=torch.ones_like(batch["labels"]),
        encoder_hidden_states=encoder_outputs.last_hidden_state,
        encoder_attention_mask=batch["attention_mask"],
    )

    assert outputs.last_hidden_state.shape == (
        batch["labels"].shape[0],
        batch["labels"].shape[1],
        model.config.hidden_size,
    )


def test_tiger_decoder_infers_missing_encoder_attention_mask() -> None:
    batch, sid_cache = _build_batch(batch_size=1)
    vocab_size = int(np.max(sid_cache) + 32)
    model = TIGERModel(_make_config(vocab_size))

    encoder_outputs = model.encoder(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )

    outputs = model.decoder(
        input_ids=batch["labels"],
        attention_mask=torch.ones_like(batch["labels"]),
        encoder_hidden_states=encoder_outputs.last_hidden_state,
        encoder_attention_mask=None,
    )

    assert outputs.last_hidden_state.shape[-1] == model.config.hidden_size


def test_tiger_model_allows_disabling_tied_embeddings() -> None:
    vocab_size = 32
    model = TIGERModel(_make_config(vocab_size, tie_word_embeddings=False))

    sentinel = []

    def recorder(self: TIGERModel, module: nn.Module, shared: nn.Module) -> None:  # type: ignore[override]
        sentinel.append((module, shared))

    model._tie_or_clone_weights = MethodType(recorder, model)
    model._tie_weights()

    assert sentinel == []
