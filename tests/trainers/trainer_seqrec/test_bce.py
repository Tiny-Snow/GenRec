from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from genrec.models.model_seqrec.base import SeqRecModelConfig, SeqRecOutput
from genrec.trainers.trainer_seqrec.bce import BCESeqRecTrainer
from tests.trainers.trainer_seqrec.helpers import (
    DummySeqRecCollator,
    DummySeqRecDataset,
    DummySeqRecModel,
    build_training_args,
)


def test_bce_seqrec_trainer_compute_rec_loss_matches_manual(tmp_path: Path) -> None:
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


def test_bce_seqrec_trainer_normalizes_embeddings_when_requested(tmp_path: Path) -> None:
    args = build_training_args(tmp_path, norm_embeddings=True)
    model = DummySeqRecModel(SeqRecModelConfig(item_size=20, hidden_size=4))
    dataset = DummySeqRecDataset(seq_len=2, num_negatives=2, item_size=model.config.item_size)
    trainer = BCESeqRecTrainer(
        model=model,
        args=args,
        data_collator=DummySeqRecCollator(),
        train_dataset=dataset,
        eval_dataset=dataset,
    )

    labels = torch.tensor([[1, 2]], dtype=torch.long)
    negative_item_ids = torch.tensor([[3, 4]], dtype=torch.long)
    attention_mask = torch.ones((1, dataset.seq_len), dtype=torch.long)
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
    assert loss.shape == torch.Size([])

    positive_emb = F.normalize(trainer.model.embed_tokens(labels), p=2, dim=-1)
    negative_emb = F.normalize(trainer.model.embed_tokens(negative_item_ids), p=2, dim=-1)
    user_emb_norm = F.normalize(user_emb, p=2, dim=-1)
    positive_scores = torch.sum(user_emb_norm * positive_emb, dim=-1)
    negative_scores = torch.matmul(user_emb_norm, negative_emb.transpose(-1, -2))
    mask = attention_mask.flatten().bool()
    positive_scores_flat = positive_scores.flatten()[mask]
    negative_scores_flat = negative_scores.reshape(-1, negative_scores.size(-1))[mask]
    expected_loss = -F.logsigmoid(positive_scores_flat).mean() - F.logsigmoid(-negative_scores_flat).mean()

    torch.testing.assert_close(loss, expected_loss)
