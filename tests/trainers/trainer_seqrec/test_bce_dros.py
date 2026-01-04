from __future__ import annotations

import copy
from pathlib import Path

import torch

from genrec.models.model_seqrec.base import SeqRecModelConfig, SeqRecOutput
from genrec.trainers.trainer_seqrec.bce import BCESeqRecTrainer
from genrec.trainers.trainer_seqrec.bce_dros import BCEDROSSeqRecTrainer, BCEDROSSeqRecTrainingArguments
from tests.trainers.trainer_seqrec.helpers import (
    DummySeqRecCollator,
    DummySeqRecDataset,
    DummySeqRecModel,
    build_training_args,
)


def test_bce_dros_seqrec_trainer_respects_dro_weight(tmp_path: Path) -> None:
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

    dros_args_zero = build_training_args(tmp_path, args_cls=BCEDROSSeqRecTrainingArguments, dros_weight=0.0)
    dros_trainer_zero = BCEDROSSeqRecTrainer(
        model=copy.deepcopy(base_model),
        args=dros_args_zero,
        data_collator=collator,
        train_dataset=dataset,
        eval_dataset=dataset,
    )
    dro_disabled_loss = dros_trainer_zero.compute_rec_loss(inputs, outputs)
    torch.testing.assert_close(dro_disabled_loss, base_loss)

    dros_args_weighted = build_training_args(tmp_path, args_cls=BCEDROSSeqRecTrainingArguments, dros_weight=0.3)
    dros_trainer_weighted = BCEDROSSeqRecTrainer(
        model=copy.deepcopy(base_model),
        args=dros_args_weighted,
        data_collator=collator,
        train_dataset=dataset,
        eval_dataset=dataset,
    )
    weighted_loss = dros_trainer_weighted.compute_rec_loss(inputs, outputs)

    assert weighted_loss >= base_loss - 1e-6
