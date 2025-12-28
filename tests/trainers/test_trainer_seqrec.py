from __future__ import annotations

import copy
from functools import partial
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset
from transformers import EvalPrediction, TrainerCallback

from genrec.models.model_seqrec.base import SeqRecModel, SeqRecModelConfig, SeqRecOutput
from genrec.trainers.trainer_seqrec.base import SeqRecTrainer, SeqRecTrainingArguments, compute_seqrec_metrics
from genrec.trainers.trainer_seqrec.bce import BCESeqRecTrainer
from genrec.trainers.trainer_seqrec.dros import DROSSeqRecTrainer, DROSSeqRecTrainingArguments
from genrec.trainers.trainer_seqrec.sl import SLSeqRecTrainer, SLSeqRecTrainingArguments
from genrec.trainers.utils.callbacks import EpochIntervalEvalCallback


class DummySeqRecModel(SeqRecModel[SeqRecModelConfig, SeqRecOutput]):
    config_class = SeqRecModelConfig

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs: Any,
    ) -> SeqRecOutput:
        embeddings = self.embed_tokens(input_ids)
        return SeqRecOutput(last_hidden_state=embeddings, model_loss=None)


class DummySeqRecDataset(Dataset):
    def __init__(self, seq_len: int = 3, num_negatives: int = 2, item_size: int = 32) -> None:
        self.seq_len = seq_len
        self.num_negatives = num_negatives
        self.length = 4
        self.item_size = item_size
        self.item_popularity = np.arange(self.item_size + 1, dtype=np.int64)
        self.train_item_popularity = self.item_popularity

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        base = idx + 1
        input_ids = torch.arange(base, base + self.seq_len, dtype=torch.long)
        attention_mask = torch.ones(self.seq_len, dtype=torch.long)
        labels = input_ids.clone()
        negative_item_ids = torch.arange(1, 1 + self.num_negatives, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "negative_item_ids": negative_item_ids,
        }


class DummySeqRecCollator:
    def __call__(self, batch: list[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        collated: Dict[str, torch.Tensor] = {}
        for key in batch[0].keys():
            collated[key] = torch.stack([sample[key] for sample in batch], dim=0)
        return collated


class DummyTrainerCallback(TrainerCallback):
    pass


class MinimalSeqRecTrainer(SeqRecTrainer[DummySeqRecModel, SeqRecTrainingArguments]):
    def compute_rec_loss(
        self,
        inputs: Dict[str, torch.Tensor],
        outputs: SeqRecOutput,
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return torch.zeros((), device=outputs.last_hidden_state.device)


class DummySeqRecModelWithOptionalLoss(DummySeqRecModel):
    def __init__(self, config: SeqRecModelConfig, model_loss_value: float | None) -> None:
        super().__init__(config)
        self._model_loss_value = model_loss_value

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs: Any,
    ) -> SeqRecOutput:
        embeddings = self.embed_tokens(input_ids)
        model_loss = None
        if self._model_loss_value is not None:
            model_loss = torch.tensor(self._model_loss_value, device=embeddings.device)
        return SeqRecOutput(last_hidden_state=embeddings, model_loss=model_loss)


class RecLossTrackingSeqRecTrainer(SeqRecTrainer[DummySeqRecModelWithOptionalLoss, SeqRecTrainingArguments]):
    def __init__(self, *args: Any, rec_loss_value: float, **kwargs: Any) -> None:
        self.rec_loss_value = rec_loss_value
        self.last_seen_num_items: torch.Tensor | None = None
        super().__init__(*args, **kwargs)

    def compute_rec_loss(
        self,
        inputs: Dict[str, torch.Tensor],
        outputs: SeqRecOutput,
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if num_items_in_batch is not None:
            self.last_seen_num_items = num_items_in_batch.clone()
        return torch.full((), self.rec_loss_value, device=outputs.last_hidden_state.device)


def build_training_args(
    base_dir: Path,
    eval_interval: int = 2,
    model_loss_weight: float = 1.0,
    metrics: Tuple[Tuple[str, Dict[str, Any]], ...] = (("hr", {}),),
    top_k: Tuple[int, ...] = (5, 10),
    *,
    args_cls: type[SeqRecTrainingArguments] = SeqRecTrainingArguments,
    **extra_kwargs: Any,
) -> SeqRecTrainingArguments:
    output_dir = base_dir / f"seqrec_trainer_tests_{args_cls.__name__.lower()}_{eval_interval}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return args_cls(
        output_dir=str(output_dir),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        use_cpu=True,
        remove_unused_columns=False,
        eval_strategy="epoch",
        logging_steps=1,
        max_steps=1,
        eval_interval=eval_interval,
        metrics=metrics,
        top_k=top_k,
        model_loss_weight=model_loss_weight,
        **extra_kwargs,
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
    assert trainer.compute_metrics.keywords["top_k"] == args.top_k

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


def test_bce_seqrec_trainer_compute_rec_loss_shapes(tmp_path: Path) -> None:
    args = build_training_args(tmp_path)
    model = DummySeqRecModel(SeqRecModelConfig(item_size=64, hidden_size=6))
    dataset = DummySeqRecDataset(seq_len=3, num_negatives=4, item_size=model.config.item_size)
    trainer = BCESeqRecTrainer(
        model=model,
        args=args,
        data_collator=DummySeqRecCollator(),
        train_dataset=dataset,
        eval_dataset=dataset,
    )

    batch_size = 2
    seq_len = 3
    num_negatives = 4
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
    labels = torch.tensor(
        [
            [1, 2, 3],
            [4, 5, 6],
        ],
        dtype=torch.long,
    )
    negative_item_ids = torch.tensor(
        [
            [10, 11, 12, 13],
            [14, 15, 16, 17],
        ],
        dtype=torch.long,
    )
    user_emb = torch.arange(batch_size * seq_len * model.config.hidden_size, dtype=torch.float32).reshape(
        batch_size,
        seq_len,
        model.config.hidden_size,
    )
    outputs = SeqRecOutput(last_hidden_state=user_emb)

    inputs = {
        "labels": labels,
        "negative_item_ids": negative_item_ids,
        "attention_mask": attention_mask,
    }

    loss = trainer.compute_rec_loss(inputs, outputs)
    assert loss.shape == torch.Size([])

    positive_emb = trainer.model.embed_tokens(labels)
    negative_emb = trainer.model.embed_tokens(negative_item_ids)
    positive_scores = torch.sum(user_emb * positive_emb, dim=-1)
    negative_scores = torch.einsum("bld,bnd->bln", user_emb, negative_emb)

    assert positive_scores.shape == (batch_size, seq_len)
    assert negative_scores.shape == (batch_size, seq_len, num_negatives)

    expected_positive_loss = -torch.nn.functional.logsigmoid(positive_scores).mean()
    expected_negative_loss = -torch.nn.functional.logsigmoid(-negative_scores).mean()
    expected_loss = expected_positive_loss + expected_negative_loss

    torch.testing.assert_close(loss, expected_loss)


def test_dros_seqrec_trainer_respects_dro_weight(tmp_path: Path) -> None:
    base_model = DummySeqRecModel(SeqRecModelConfig(item_size=32, hidden_size=4))
    dataset = DummySeqRecDataset(seq_len=3, num_negatives=2, item_size=base_model.config.item_size)
    collator = DummySeqRecCollator()

    attention_mask = torch.ones((2, dataset.seq_len), dtype=torch.long)
    labels = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    negative_item_ids = torch.tensor([[7, 8], [9, 10]], dtype=torch.long)
    user_emb = torch.arange(2 * dataset.seq_len * base_model.config.hidden_size, dtype=torch.float32).reshape(
        2,
        dataset.seq_len,
        base_model.config.hidden_size,
    )
    outputs = SeqRecOutput(last_hidden_state=user_emb)

    inputs = {
        "labels": labels,
        "negative_item_ids": negative_item_ids,
        "attention_mask": attention_mask,
    }

    bce_args = build_training_args(tmp_path)
    bce_trainer = BCESeqRecTrainer(
        model=base_model,
        args=bce_args,
        data_collator=collator,
        train_dataset=dataset,
        eval_dataset=dataset,
    )
    base_loss = bce_trainer.compute_rec_loss(inputs, outputs)

    dros_args_zero = build_training_args(tmp_path, args_cls=DROSSeqRecTrainingArguments, dros_weight=0.0)
    dros_trainer_zero = DROSSeqRecTrainer(
        model=copy.deepcopy(base_model),
        args=dros_args_zero,
        data_collator=collator,
        train_dataset=dataset,
        eval_dataset=dataset,
    )
    dro_disabled_loss = dros_trainer_zero.compute_rec_loss(inputs, outputs)
    torch.testing.assert_close(dro_disabled_loss, base_loss)

    dros_args_weighted = build_training_args(tmp_path, args_cls=DROSSeqRecTrainingArguments, dros_weight=0.3)
    dros_trainer_weighted = DROSSeqRecTrainer(
        model=copy.deepcopy(base_model),
        args=dros_args_weighted,
        data_collator=collator,
        train_dataset=dataset,
        eval_dataset=dataset,
    )
    weighted_loss = dros_trainer_weighted.compute_rec_loss(inputs, outputs)

    assert weighted_loss >= base_loss - 1e-6


def test_sl_seqrec_trainer_compute_rec_loss_matches_manual(tmp_path: Path) -> None:
    args = build_training_args(tmp_path, args_cls=SLSeqRecTrainingArguments, sl_temperature=0.5)
    model = DummySeqRecModel(SeqRecModelConfig(item_size=40, hidden_size=4))
    dataset = DummySeqRecDataset(seq_len=2, num_negatives=3, item_size=model.config.item_size)
    trainer = SLSeqRecTrainer(
        model=model,
        args=args,
        data_collator=DummySeqRecCollator(),
        train_dataset=dataset,
        eval_dataset=dataset,
    )

    attention_mask = torch.ones((2, dataset.seq_len), dtype=torch.long)
    labels = torch.tensor([[1, 2], [3, 4]], dtype=torch.long)
    negative_item_ids = torch.tensor([[5, 6, 7], [8, 9, 10]], dtype=torch.long)
    user_emb = torch.arange(2 * dataset.seq_len * model.config.hidden_size, dtype=torch.float32).reshape(
        2,
        dataset.seq_len,
        model.config.hidden_size,
    )
    outputs = SeqRecOutput(last_hidden_state=user_emb)

    inputs = {
        "labels": labels,
        "negative_item_ids": negative_item_ids,
        "attention_mask": attention_mask,
    }

    loss = trainer.compute_rec_loss(inputs, outputs)

    positive_emb = trainer.model.embed_tokens(labels)
    negative_emb = trainer.model.embed_tokens(negative_item_ids)
    positive_scores = torch.sum(user_emb * positive_emb, dim=-1)
    negative_scores = torch.matmul(user_emb, negative_emb.transpose(-1, -2))
    mask = attention_mask.flatten().bool()
    positive_scores_flat = positive_scores.flatten()[mask]
    negative_scores_flat = negative_scores.reshape(-1, negative_scores.size(-1))[mask]
    diff_scores = negative_scores_flat - positive_scores_flat.unsqueeze(-1)
    tau = args.sl_temperature
    expected_loss = (torch.logsumexp(diff_scores / tau, dim=-1) * tau).mean()

    torch.testing.assert_close(loss, expected_loss)


def test_compute_seqrec_metrics_returns_registered_metrics() -> None:
    logits = np.array(
        [
            [0.2, 0.8, 0.1],
            [0.9, 0.05, 0.6],
        ],
        dtype=np.float32,
    )
    labels = np.array([[0, 1], [1, 2]], dtype=np.int64)
    prediction = EvalPrediction(predictions=logits, label_ids=labels)

    class TinyTrainDataset:
        def __init__(self, vocab_size: int) -> None:
            self.item_popularity = np.ones(vocab_size + 1, dtype=np.int64)

    train_dataset = TinyTrainDataset(vocab_size=logits.shape[1] - 1)
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
    assert output_dict["logits"].shape == (len(batch), model.config.item_size + 1)
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
    assert "logits" in outputs and outputs["logits"].shape == (
        len(batch),
        model.config.item_size + 1,
    )


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
    expected_shape = (len(dataset), model.config.item_size + 1)
    assert prediction_output.predictions.shape == expected_shape
    assert prediction_output.label_ids.shape == (len(dataset), dataset.seq_len)
    assert "test_hr@1" in prediction_output.metrics
    assert 0.0 <= prediction_output.metrics["test_hr@1"] <= 1.0


def test_bce_seqrec_trainer_requires_negative_items(tmp_path: Path) -> None:
    args = build_training_args(tmp_path)
    model = DummySeqRecModel(SeqRecModelConfig(item_size=16, hidden_size=4))
    dataset = DummySeqRecDataset(item_size=model.config.item_size)
    trainer = BCESeqRecTrainer(
        model=model,
        args=args,
        data_collator=DummySeqRecCollator(),
        train_dataset=dataset,
        eval_dataset=dataset,
    )

    inputs = {
        "labels": torch.ones((1, dataset.seq_len), dtype=torch.long),
        "negative_item_ids": None,
    }
    outputs = SeqRecOutput(
        last_hidden_state=torch.zeros((1, dataset.seq_len, model.config.hidden_size), dtype=torch.float32)
    )

    with pytest.raises(AssertionError):
        trainer.compute_rec_loss(inputs, outputs)
