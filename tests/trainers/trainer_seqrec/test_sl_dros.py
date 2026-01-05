from __future__ import annotations

import copy
from pathlib import Path

import torch

from genrec.models.model_seqrec.base import SeqRecModelConfig, SeqRecOutput
from genrec.trainers.trainer_seqrec.sl import SLSeqRecTrainer, SLSeqRecTrainingArguments
from genrec.trainers.trainer_seqrec.sl_dros import SLDROSSeqRecTrainer, SLDROSSeqRecTrainingArguments
from tests.trainers.trainer_seqrec.helpers import (
    DummySeqRecCollator,
    DummySeqRecDataset,
    DummySeqRecModel,
    build_training_args,
)


def test_sl_dros_seqrec_trainer_respects_dro_weight(tmp_path: Path) -> None:
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

    sl_args = build_training_args(
        tmp_path,
        args_cls=SLSeqRecTrainingArguments,
        sl_temperature=0.4,
        stepwise_negative_sampling=False,
    )
    sl_trainer = SLSeqRecTrainer(
        model=base_model,
        args=sl_args,
        data_collator=collator,
        train_dataset=dataset,
        eval_dataset=dataset,
    )
    base_loss = sl_trainer.compute_rec_loss(inputs, outputs)

    dros_args_zero = build_training_args(
        tmp_path,
        args_cls=SLDROSSeqRecTrainingArguments,
        sl_temperature=0.4,
        dros_weight=0.0,
        stepwise_negative_sampling=False,
    )
    dros_trainer_zero = SLDROSSeqRecTrainer(
        model=copy.deepcopy(base_model),
        args=dros_args_zero,
        data_collator=collator,
        train_dataset=dataset,
        eval_dataset=dataset,
    )
    dro_disabled_loss = dros_trainer_zero.compute_rec_loss(inputs, outputs)
    torch.testing.assert_close(dro_disabled_loss, base_loss)

    dros_args_weighted = build_training_args(
        tmp_path,
        args_cls=SLDROSSeqRecTrainingArguments,
        sl_temperature=0.4,
        dros_weight=0.25,
        stepwise_negative_sampling=False,
    )
    dros_trainer_weighted = SLDROSSeqRecTrainer(
        model=copy.deepcopy(base_model),
        args=dros_args_weighted,
        data_collator=collator,
        train_dataset=dataset,
        eval_dataset=dataset,
    )
    weighted_loss = dros_trainer_weighted.compute_rec_loss(inputs, outputs)

    assert weighted_loss >= base_loss - 1e-6


def test_sl_dros_seqrec_trainer_normalization_branch(tmp_path: Path) -> None:
    args = build_training_args(
        tmp_path,
        args_cls=SLDROSSeqRecTrainingArguments,
        sl_temperature=0.35,
        dros_weight=0.1,
        norm_embeddings=True,
        stepwise_negative_sampling=False,
    )
    model = DummySeqRecModel(SeqRecModelConfig(item_size=20, hidden_size=4))
    dataset = DummySeqRecDataset(seq_len=2, num_negatives=2, item_size=model.config.item_size)
    trainer = SLDROSSeqRecTrainer(
        model=model,
        args=args,
        data_collator=DummySeqRecCollator(),
        train_dataset=dataset,
        eval_dataset=dataset,
    )

    batch = [dataset[0], dataset[1]]
    inputs = trainer.data_collator(batch)
    outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])

    loss = trainer.compute_rec_loss(inputs, outputs, norm_embeddings=True)
    assert torch.isfinite(loss).item()


def test_sl_dros_seqrec_trainer_stepwise_negative_sampling_branch(tmp_path: Path) -> None:
    torch.manual_seed(0)
    args = build_training_args(
        tmp_path,
        args_cls=SLDROSSeqRecTrainingArguments,
        sl_temperature=0.3,
        dros_weight=0.1,
        stepwise_negative_sampling=True,
        norm_embeddings=True,
    )
    model = DummySeqRecModel(SeqRecModelConfig(item_size=22, hidden_size=4))
    dataset = DummySeqRecDataset(seq_len=2, num_negatives=2, item_size=model.config.item_size)
    trainer = SLDROSSeqRecTrainer(
        model=model,
        args=args,
        data_collator=DummySeqRecCollator(),
        train_dataset=dataset,
        eval_dataset=dataset,
    )

    batch = [dataset[0], dataset[1]]
    inputs = trainer.data_collator(batch)
    outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])

    loss = trainer.compute_rec_loss(inputs, outputs)
    loss_norm = trainer.compute_rec_loss(inputs, outputs, norm_embeddings=True)
    assert torch.isfinite(loss).item()
    assert torch.isfinite(loss_norm).item()
