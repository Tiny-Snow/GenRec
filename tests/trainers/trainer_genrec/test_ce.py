from __future__ import annotations

import torch
import torch.nn.functional as F

from genrec.models.model_genrec.base import GenRecModelConfig
from genrec.trainers.trainer_genrec.ce import CEGenRecTrainer, CEGenRecTrainingArguments
from tests.trainers.trainer_genrec.helpers import (
    DummyGenRecCollator,
    DummyGenRecDataset,
    DummyGenRecModel,
    build_genrec_training_args,
)


def test_ce_genrec_trainer_compute_rec_loss_matches_cross_entropy(tmp_path):
    dataset = DummyGenRecDataset(length=2)
    args = build_genrec_training_args(tmp_path, args_cls=CEGenRecTrainingArguments)
    model = DummyGenRecModel(
        GenRecModelConfig(vocab_size=32, hidden_size=8, decoder_start_token_id=0, pad_token_id=0, use_cache=True),
        sid_width=dataset.sid_width,
    )
    trainer = CEGenRecTrainer(
        model=model,
        args=args,
        data_collator=DummyGenRecCollator(),
        train_dataset=dataset,
        eval_dataset=dataset,
    )

    batch = trainer.data_collator([dataset[0], dataset[1]])
    outputs = trainer.model(**batch)

    ce_loss = trainer.compute_rec_loss(batch, outputs)

    logits = outputs.logits.view(-1, outputs.logits.size(-1))
    labels = batch["labels"].view(-1)
    expected_loss = F.cross_entropy(logits, labels)
    torch.testing.assert_close(ce_loss, expected_loss)
