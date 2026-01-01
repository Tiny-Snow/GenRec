from __future__ import annotations

from pathlib import Path

import torch

from genrec.models.model_seqrec.base import SeqRecModelConfig
from genrec.trainers.trainer_seqrec.resn import ReSNSeqRecTrainer, ReSNSeqRecTrainingArguments
from tests.trainers.trainer_seqrec.helpers import (
    DummySeqRecCollator,
    DummySeqRecDataset,
    DummySeqRecModel,
    build_training_args,
)


def test_resn_seqrec_trainer_backward_pass(tmp_path: Path) -> None:
    args = build_training_args(tmp_path, args_cls=ReSNSeqRecTrainingArguments, resn_weight=0.5)
    model = DummySeqRecModel(SeqRecModelConfig(item_size=24, hidden_size=4))
    dataset = DummySeqRecDataset(seq_len=3, num_negatives=2, item_size=model.config.item_size)
    trainer = ReSNSeqRecTrainer(
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
