from __future__ import annotations

import copy
from functools import partial
from pathlib import Path
from typing import Dict

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from transformers import EvalPrediction

from genrec.models.model_seqrec.base import SeqRecModelConfig
from genrec.trainers.trainer_seqrec.base import compute_seqrec_metrics
from genrec.trainers.utils.evaluations import clip_top_k
from genrec.trainers.utils.callbacks import EpochIntervalEvalCallback
from tests.trainers.trainer_seqrec.helpers import (
    DummySeqRecCollator,
    DummySeqRecDataset,
    DummySeqRecModel,
    DummySeqRecModelWithOptionalLoss,
    DummyTrainerCallback,
    MinimalSeqRecTrainer,
    RecLossTrackingSeqRecTrainer,
    build_training_args,
)


def test_seqrec_trainer_initializes_defaults(tmp_path: Path) -> None:
    args = build_training_args(tmp_path, eval_interval=3)
    model = DummySeqRecModel(SeqRecModelConfig(item_size=32, hidden_size=8))
    dataset = DummySeqRecDataset(item_size=model.config.item_size)
    collator = DummySeqRecCollator()

    trainer = MinimalSeqRecTrainer(
        model=model,
        args=args,
        data_collator=collator,
        train_dataset=dataset,
        eval_dataset=dataset,
    )

    assert isinstance(trainer.compute_metrics, partial)
    assert trainer.compute_metrics.func is compute_seqrec_metrics
    assert trainer.compute_metrics.keywords["metrics"] == args.metrics
    expected_top_k = tuple(clip_top_k(args.top_k, dataset.item_size))
    assert trainer.compute_metrics.keywords["top_k"] == expected_top_k
    assert trainer.top_k == expected_top_k
    assert trainer.max_top_k == max(expected_top_k)

    callback_types = tuple(type(cb) for cb in trainer.callback_handler.callbacks)
    assert any(isinstance(cb, EpochIntervalEvalCallback) for cb in trainer.callback_handler.callbacks), callback_types


def test_seqrec_trainer_respects_custom_metrics_and_callbacks(tmp_path: Path) -> None:
    args = build_training_args(tmp_path)
    model = DummySeqRecModel(SeqRecModelConfig(item_size=16, hidden_size=4))
    dataset = DummySeqRecDataset(item_size=model.config.item_size)
    collator = DummySeqRecCollator()

    def custom_metrics(prediction: EvalPrediction) -> Dict[str, float]:
        return {"custom": 0.0}

    custom_callback = DummyTrainerCallback()

    trainer = MinimalSeqRecTrainer(
        model=model,
        args=args,
        data_collator=collator,
        train_dataset=dataset,
        eval_dataset=dataset,
        compute_metrics=custom_metrics,
        callbacks=[custom_callback],
    )

    assert trainer.compute_metrics is custom_metrics
    assert any(isinstance(cb, DummyTrainerCallback) for cb in trainer.callback_handler.callbacks)


def test_compute_seqrec_metrics_returns_registered_metrics() -> None:
    topk_indices = np.array(
        [
            [0, 1],
            [2, 1],
        ],
        dtype=np.int64,
    )
    labels = np.array([[0, 1], [1, 2]], dtype=np.int64)
    prediction = EvalPrediction(predictions=topk_indices, label_ids=labels)

    class TinyTrainDataset:
        def __init__(self, vocab_size: int) -> None:
            self.item_popularity = np.ones(vocab_size + 1, dtype=np.int64)

    train_dataset = TinyTrainDataset(vocab_size=int(topk_indices.max()))
    metrics = compute_seqrec_metrics(
        prediction,
        train_dataset=train_dataset,
        top_k=(1, 2),
        metrics=(("hr", {}), ("ndcg", {})),
    )

    expected_values = {
        "hr@1": 0.5,
        "hr@2": 1.0,
        "ndcg@1": 0.5,
        "ndcg@2": (1.0 + (1.0 / np.log2(3))) / 2,
    }
    for key, expected in expected_values.items():
        assert metrics[key] == pytest.approx(expected, rel=1e-6)


def test_seqrec_trainer_compute_loss_with_model_loss_and_outputs(
    tmp_path: Path,
) -> None:
    args = build_training_args(tmp_path, eval_interval=1, model_loss_weight=0.3)
    model = DummySeqRecModelWithOptionalLoss(SeqRecModelConfig(item_size=8, hidden_size=4), model_loss_value=2.0)
    dataset = DummySeqRecDataset(seq_len=3, num_negatives=2, item_size=model.config.item_size)
    collator = DummySeqRecCollator()
    trainer = RecLossTrackingSeqRecTrainer(
        model=model,
        args=args,
        data_collator=collator,
        train_dataset=dataset,
        eval_dataset=dataset,
        rec_loss_value=1.5,
    )

    batch = [dataset[0], dataset[1]]
    inputs = collator(batch)
    num_items_in_batch = torch.tensor([dataset.seq_len, dataset.seq_len], dtype=torch.long)

    loss, output_dict = trainer.compute_loss(
        trainer.model,
        inputs,
        return_outputs=True,
        num_items_in_batch=num_items_in_batch,
    )

    expected_loss = 1.5 + 2.0 * args.model_loss_weight
    torch.testing.assert_close(loss, torch.tensor(expected_loss))
    torch.testing.assert_close(output_dict["loss"], loss)
    assert "topk_indices" in output_dict
    assert output_dict["topk_indices"].shape == (len(batch), trainer.max_top_k)
    forward_outputs = trainer.model(**inputs)
    last_hidden = forward_outputs.last_hidden_state[:, -1, :]
    item_embed_weight = trainer.model.item_embed_weight
    expected_logits = last_hidden @ item_embed_weight.T
    _, expected_topk = torch.topk(expected_logits[:, 1:], k=trainer.max_top_k, dim=1)
    torch.testing.assert_close(output_dict["topk_indices"], expected_topk)
    assert torch.equal(trainer.last_seen_num_items, num_items_in_batch)


def test_seqrec_trainer_compute_loss_without_model_loss_returns_rec_loss(
    tmp_path: Path,
) -> None:
    args = build_training_args(tmp_path, eval_interval=1)
    model = DummySeqRecModelWithOptionalLoss(SeqRecModelConfig(item_size=6, hidden_size=4), model_loss_value=None)
    dataset = DummySeqRecDataset(seq_len=2, num_negatives=1, item_size=model.config.item_size)
    collator = DummySeqRecCollator()
    trainer = RecLossTrackingSeqRecTrainer(
        model=model,
        args=args,
        data_collator=collator,
        train_dataset=dataset,
        eval_dataset=dataset,
        rec_loss_value=0.75,
    )

    batch = [dataset[0]]
    inputs = collator(batch)

    loss_only = trainer.compute_loss(trainer.model, inputs)
    assert trainer.last_seen_num_items is None
    torch.testing.assert_close(loss_only, torch.tensor(0.75))


def test_seqrec_trainer_compute_loss_with_model_loss_without_outputs(
    tmp_path: Path,
) -> None:
    args = build_training_args(tmp_path, eval_interval=1, model_loss_weight=0.2)
    model = DummySeqRecModelWithOptionalLoss(SeqRecModelConfig(item_size=6, hidden_size=4), model_loss_value=3.0)
    dataset = DummySeqRecDataset(seq_len=2, num_negatives=1, item_size=model.config.item_size)
    collator = DummySeqRecCollator()
    trainer = RecLossTrackingSeqRecTrainer(
        model=model,
        args=args,
        data_collator=collator,
        train_dataset=dataset,
        eval_dataset=dataset,
        rec_loss_value=1.0,
    )

    batch = [dataset[0]]
    inputs = collator(batch)

    loss_only = trainer.compute_loss(trainer.model, inputs, return_outputs=False)
    expected = 1.0 + 3.0 * args.model_loss_weight
    torch.testing.assert_close(loss_only, torch.tensor(expected))


def test_seqrec_trainer_compute_loss_without_model_loss_returns_outputs(
    tmp_path: Path,
) -> None:
    args = build_training_args(tmp_path, eval_interval=1)
    model = DummySeqRecModelWithOptionalLoss(SeqRecModelConfig(item_size=5, hidden_size=3), model_loss_value=None)
    dataset = DummySeqRecDataset(seq_len=2, num_negatives=1, item_size=model.config.item_size)
    collator = DummySeqRecCollator()
    trainer = RecLossTrackingSeqRecTrainer(
        model=model,
        args=args,
        data_collator=collator,
        train_dataset=dataset,
        eval_dataset=dataset,
        rec_loss_value=0.6,
    )

    batch = [dataset[0]]
    inputs = collator(batch)

    loss, outputs = trainer.compute_loss(trainer.model, inputs, return_outputs=True)
    torch.testing.assert_close(loss, torch.tensor(0.6))
    assert "topk_indices" in outputs
    assert outputs["topk_indices"].shape == (len(batch), trainer.max_top_k)


def test_seqrec_trainer_normalizes_logits_when_enabled(tmp_path: Path) -> None:
    args = build_training_args(tmp_path, eval_interval=1, norm_embeddings=True)
    model = DummySeqRecModel(SeqRecModelConfig(item_size=12, hidden_size=4))
    dataset = DummySeqRecDataset(seq_len=3, num_negatives=2, item_size=model.config.item_size)
    trainer = MinimalSeqRecTrainer(
        model=model,
        args=args,
        data_collator=DummySeqRecCollator(),
        train_dataset=dataset,
        eval_dataset=dataset,
    )

    batch = [dataset[0], dataset[1]]
    inputs = trainer.data_collator(batch)

    loss, output_dict = trainer.compute_loss(trainer.model, inputs, return_outputs=True)
    torch.testing.assert_close(loss, torch.zeros((), device=loss.device))

    forward_outputs = trainer.model(**inputs)
    last_hidden = forward_outputs.last_hidden_state[:, -1, :]
    item_embed_weight = trainer.model.item_embed_weight
    expected_logits = F.normalize(last_hidden, p=2, dim=-1) @ F.normalize(item_embed_weight, p=2, dim=-1).T
    expected_logits = expected_logits[:, 1:]  # exclude padding token logits

    _, expected_topk = torch.topk(expected_logits, k=trainer.max_top_k, dim=1)
    torch.testing.assert_close(output_dict["topk_indices"], expected_topk)


def test_seqrec_trainer_predict_returns_metrics(tmp_path: Path) -> None:
    args = build_training_args(tmp_path, eval_interval=1, metrics=(("hr", {}),), top_k=(1,))
    model = DummySeqRecModel(SeqRecModelConfig(item_size=24, hidden_size=4))
    dataset = DummySeqRecDataset(seq_len=3, num_negatives=2, item_size=model.config.item_size)
    trainer = MinimalSeqRecTrainer(
        model=model,
        args=args,
        data_collator=DummySeqRecCollator(),
        train_dataset=dataset,
        eval_dataset=dataset,
    )

    prediction_output = trainer.predict(dataset)
    expected_shape = (len(dataset), trainer.max_top_k)
    assert prediction_output.predictions.shape == expected_shape
    assert prediction_output.label_ids.shape == (len(dataset), dataset.seq_len)
    assert "test_hr@1" in prediction_output.metrics
    assert 0.0 <= prediction_output.metrics["test_hr@1"] <= 1.0


def test_sanitize_top_k_clamps_values_exceeding_item_size() -> None:
    result = clip_top_k((1, 5, 10, 50, 50), item_size=10)
    assert result == (1, 5, 10)
