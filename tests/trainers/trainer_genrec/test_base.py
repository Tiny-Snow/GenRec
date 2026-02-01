from __future__ import annotations

from functools import partial
from pathlib import Path

import torch

from genrec.models.model_genrec.base import GenRecModelConfig
from genrec.trainers.trainer_genrec.base import compute_genrec_metrics
from genrec.trainers.trainer_genrec.utils.callbacks import EpochIntervalEvalCallback, HardStopCallback
from genrec.trainers.trainer_genrec.utils.evaluations import clip_top_k
from tests.trainers.trainer_genrec.helpers import (
    DummyGenRecCollator,
    DummyGenRecDataset,
    DummyGenRecModel,
    DummyTrainerCallback,
    MinimalGenRecTrainer,
    RecLossTrackingGenRecTrainer,
    build_genrec_training_args,
)


def _build_model(dataset: DummyGenRecDataset, model_loss_value: float | None = None) -> DummyGenRecModel:
    config = GenRecModelConfig(
        vocab_size=64,
        hidden_size=16,
        decoder_start_token_id=0,
        pad_token_id=0,
        use_cache=True,
    )
    return DummyGenRecModel(config, sid_width=dataset.sid_width, model_loss_value=model_loss_value)


def test_genrec_trainer_initializes_defaults(tmp_path: Path) -> None:
    dataset = DummyGenRecDataset(item_size=10, sid_width=3)
    args = build_genrec_training_args(tmp_path, eval_interval=3, top_k=(1, 4), num_beams=4)
    trainer = MinimalGenRecTrainer(
        model=_build_model(dataset),
        args=args,
        data_collator=DummyGenRecCollator(),
        train_dataset=dataset,
        eval_dataset=dataset,
    )

    assert isinstance(trainer.compute_metrics, partial)
    assert trainer.compute_metrics.func is compute_genrec_metrics
    expected_top_k = tuple(clip_top_k(args.top_k, dataset.item_size))
    assert trainer.top_k == expected_top_k
    assert trainer.max_top_k == max(expected_top_k)
    assert trainer.gen_cfg.max_new_tokens == dataset.sid_width
    assert trainer.num_beams == args.num_beams
    assert trainer.prefix_allowed_tokens_fn is not None

    callbacks = trainer.callback_handler.callbacks
    assert any(isinstance(cb, EpochIntervalEvalCallback) for cb in callbacks)
    assert any(isinstance(cb, HardStopCallback) for cb in callbacks)


def test_genrec_trainer_respects_custom_metrics_and_callbacks(tmp_path: Path) -> None:
    dataset = DummyGenRecDataset(item_size=6)
    args = build_genrec_training_args(tmp_path)

    def custom_metrics(*_args, **_kwargs):
        return {"custom": 1.0}

    custom_callback = DummyTrainerCallback()

    trainer = MinimalGenRecTrainer(
        model=_build_model(dataset),
        args=args,
        data_collator=DummyGenRecCollator(),
        train_dataset=dataset,
        eval_dataset=dataset,
        compute_metrics=custom_metrics,
        callbacks=[custom_callback],
    )

    assert trainer.compute_metrics is custom_metrics
    assert custom_callback in trainer.callback_handler.callbacks


def test_genrec_trainer_compute_loss_with_model_loss(tmp_path: Path) -> None:
    dataset = DummyGenRecDataset(length=2)
    args = build_genrec_training_args(tmp_path, eval_interval=1, model_loss_weight=0.25)
    trainer = RecLossTrackingGenRecTrainer(
        model=_build_model(dataset, model_loss_value=2.0),
        args=args,
        data_collator=DummyGenRecCollator(),
        train_dataset=dataset,
        eval_dataset=dataset,
        rec_loss_value=1.5,
    )

    batch = trainer.data_collator([dataset[0], dataset[1]])
    num_items = torch.full((batch["input_ids"].size(0),), dataset.sid_width, dtype=torch.long)

    loss, outputs = trainer.compute_loss(
        trainer.model,
        batch,
        return_outputs=True,
        num_items_in_batch=num_items,
    )

    expected_loss = 1.5 + 2.0 * args.model_loss_weight
    torch.testing.assert_close(loss, torch.tensor(expected_loss))
    assert outputs is not None and outputs.logits is not None
    assert torch.equal(trainer.last_seen_num_items, num_items)


def test_genrec_trainer_compute_loss_without_model_loss(tmp_path: Path) -> None:
    dataset = DummyGenRecDataset(length=1)
    args = build_genrec_training_args(tmp_path, eval_interval=1)
    trainer = RecLossTrackingGenRecTrainer(
        model=_build_model(dataset, model_loss_value=None),
        args=args,
        data_collator=DummyGenRecCollator(),
        train_dataset=dataset,
        eval_dataset=dataset,
        rec_loss_value=0.8,
    )

    batch = trainer.data_collator([dataset[0]])
    loss_only = trainer.compute_loss(trainer.model, batch, return_outputs=False)

    torch.testing.assert_close(loss_only, torch.tensor(0.8))
    assert trainer.last_seen_num_items is None


def test_genrec_trainer_prediction_step_returns_sequences(tmp_path: Path) -> None:
    dataset = DummyGenRecDataset(length=2)
    args = build_genrec_training_args(tmp_path, eval_interval=1, top_k=(1, 2), num_beams=2)
    trainer = MinimalGenRecTrainer(
        model=_build_model(dataset),
        args=args,
        data_collator=DummyGenRecCollator(),
        train_dataset=dataset,
        eval_dataset=dataset,
    )

    batch = trainer.data_collator([dataset[0], dataset[1]])

    loss, predictions, labels = trainer.prediction_step(
        trainer.model,
        batch,
        prediction_loss_only=False,
    )

    assert loss is not None
    assert predictions is not None
    assert labels is not None
    assert predictions.shape == (batch["input_ids"].size(0), trainer.num_beams, trainer.sid_width)
    assert labels.shape == (batch["input_ids"].size(0), trainer.sid_width)

    expected_sequences = []
    for batch_idx in range(batch["input_ids"].size(0)):
        beam_rows = []
        for beam_idx in range(trainer.num_beams):
            start = batch_idx * trainer.num_beams + beam_idx + 1
            beam_rows.append(torch.arange(start, start + trainer.sid_width, dtype=torch.long))
        expected_sequences.append(torch.stack(beam_rows, dim=0))
    expected = torch.stack(expected_sequences, dim=0)
    torch.testing.assert_close(predictions, expected)
