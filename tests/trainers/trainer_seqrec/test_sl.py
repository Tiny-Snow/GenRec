from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from genrec.models.model_seqrec.base import SeqRecModelConfig, SeqRecOutput
from genrec.trainers.trainer_seqrec.sl import SLSeqRecTrainer, SLSeqRecTrainingArguments
from tests.trainers.trainer_seqrec.helpers import (
    DummySeqRecCollator,
    DummySeqRecDataset,
    DummySeqRecModel,
    build_training_args,
)


def test_sl_seqrec_trainer_compute_rec_loss_matches_manual(tmp_path: Path) -> None:
    args = build_training_args(
        tmp_path,
        args_cls=SLSeqRecTrainingArguments,
        sl_temperature=0.5,
        stepwise_negative_sampling=False,
    )
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
    tau = args.sl_temperature
    all_scores = torch.cat([positive_scores_flat.unsqueeze(-1), negative_scores_flat], dim=-1)
    expected_loss = -F.log_softmax(all_scores / tau, dim=-1)[:, 0].mean()

    torch.testing.assert_close(loss, expected_loss)


def test_sl_seqrec_trainer_normalization_branch(tmp_path: Path) -> None:
    args = build_training_args(
        tmp_path,
        args_cls=SLSeqRecTrainingArguments,
        sl_temperature=0.2,
        norm_embeddings=True,
        stepwise_negative_sampling=False,
    )
    model = DummySeqRecModel(SeqRecModelConfig(item_size=30, hidden_size=4))
    dataset = DummySeqRecDataset(seq_len=2, num_negatives=2, item_size=model.config.item_size)
    trainer = SLSeqRecTrainer(
        model=model,
        args=args,
        data_collator=DummySeqRecCollator(),
        train_dataset=dataset,
        eval_dataset=dataset,
    )

    attention_mask = torch.ones((1, dataset.seq_len), dtype=torch.long)
    labels = torch.tensor([[1, 2]], dtype=torch.long)
    negative_item_ids = torch.tensor([[3, 4]], dtype=torch.long)
    user_emb = torch.arange(dataset.seq_len * model.config.hidden_size, dtype=torch.float32).reshape(
        1,
        dataset.seq_len,
        model.config.hidden_size,
    )
    outputs = SeqRecOutput(last_hidden_state=user_emb)
    inputs = {
        "labels": labels,
        "negative_item_ids": negative_item_ids,
        "attention_mask": attention_mask,
    }

    loss = trainer.compute_rec_loss(inputs, outputs, norm_embeddings=True)

    user_emb_norm = F.normalize(user_emb, p=2, dim=-1)
    positive_emb = F.normalize(trainer.model.embed_tokens(labels), p=2, dim=-1)
    negative_emb = F.normalize(trainer.model.embed_tokens(negative_item_ids), p=2, dim=-1)
    positive_scores = torch.sum(user_emb_norm * positive_emb, dim=-1)
    negative_scores = torch.matmul(user_emb_norm, negative_emb.transpose(-1, -2))
    mask = attention_mask.flatten().bool()
    positive_scores_flat = positive_scores.flatten()[mask]
    negative_scores_flat = negative_scores.reshape(-1, negative_scores.size(-1))[mask]
    all_scores = torch.cat([positive_scores_flat.unsqueeze(-1), negative_scores_flat], dim=-1)
    expected = -F.log_softmax(all_scores / args.sl_temperature, dim=-1)[:, 0].mean()

    torch.testing.assert_close(loss, expected)


def test_sl_seqrec_trainer_stepwise_negative_sampling_branch(tmp_path: Path) -> None:
    torch.manual_seed(0)
    args = build_training_args(
        tmp_path,
        args_cls=SLSeqRecTrainingArguments,
        sl_temperature=0.3,
        stepwise_negative_sampling=True,
        norm_embeddings=True,
    )
    model = DummySeqRecModel(SeqRecModelConfig(item_size=28, hidden_size=4))
    dataset = DummySeqRecDataset(seq_len=3, num_negatives=2, item_size=model.config.item_size)
    trainer = SLSeqRecTrainer(
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
