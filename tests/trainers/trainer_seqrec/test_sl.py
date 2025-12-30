from __future__ import annotations

from pathlib import Path

import torch

from genrec.models.model_seqrec.base import SeqRecModelConfig, SeqRecOutput
from genrec.trainers.trainer_seqrec.sl import SLSeqRecTrainer, SLSeqRecTrainingArguments
from tests.trainers.trainer_seqrec.helpers import (
    DummySeqRecCollator,
    DummySeqRecDataset,
    DummySeqRecModel,
    build_training_args,
)


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
