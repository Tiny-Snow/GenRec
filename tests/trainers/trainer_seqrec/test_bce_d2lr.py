from __future__ import annotations

from pathlib import Path

import torch

from genrec.models.model_seqrec.base import SeqRecModelConfig
from genrec.trainers.trainer_seqrec.bce_d2lr import BCED2LRSeqRecTrainer, BCED2LRSeqRecTrainingArguments
from tests.trainers.trainer_seqrec.helpers import (
    DummySeqRecCollator,
    DummySeqRecDataset,
    DummySeqRecModel,
    build_training_args,
)


def test_bce_d2lr_seqrec_trainer_backward_pass(tmp_path: Path) -> None:
    args = build_training_args(
        tmp_path,
        args_cls=BCED2LRSeqRecTrainingArguments,
        d2lr_ips_temperature=0.5,
    )
    model = DummySeqRecModel(SeqRecModelConfig(item_size=24, hidden_size=4))
    dataset = DummySeqRecDataset(seq_len=3, num_negatives=2, item_size=model.config.item_size)
    trainer = BCED2LRSeqRecTrainer(
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

    assert torch.isfinite(loss).item()

    model.zero_grad()
    loss.backward()

    assert model.item_embed.weight.grad is not None
    assert torch.count_nonzero(model.item_embed.weight.grad).item() > 0


def test_bce_d2lr_seqrec_trainer_normalization_branch(tmp_path: Path) -> None:
    args = build_training_args(
        tmp_path,
        args_cls=BCED2LRSeqRecTrainingArguments,
        d2lr_ips_temperature=0.3,
        norm_embeddings=True,
    )
    model = DummySeqRecModel(SeqRecModelConfig(item_size=16, hidden_size=4))
    dataset = DummySeqRecDataset(seq_len=2, num_negatives=2, item_size=model.config.item_size)
    trainer = BCED2LRSeqRecTrainer(
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
