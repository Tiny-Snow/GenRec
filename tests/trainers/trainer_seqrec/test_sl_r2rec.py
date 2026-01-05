from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from genrec.models.model_seqrec.base import SeqRecModelConfig, SeqRecOutput
from genrec.trainers.trainer_seqrec.sl_r2rec import SLR2RecSeqRecTrainer, SLR2RecSeqRecTrainingArguments
from tests.trainers.trainer_seqrec.helpers import (
    DummySeqRecCollator,
    DummySeqRecDataset,
    DummySeqRecModel,
    build_training_args,
)


def _build_trainer(tmp_path: Path, item_size: int = 16) -> SLR2RecSeqRecTrainer:
    args = build_training_args(
        tmp_path,
        args_cls=SLR2RecSeqRecTrainingArguments,
        sl_temperature=0.35,
        tail_item_ratio=0.5,
        r2rec_tau=1.0,
        r2rec_alpha_b=2.0,
        r2rec_alpha_p=0.0,
    )
    model = DummySeqRecModel(SeqRecModelConfig(item_size=item_size, hidden_size=4))
    dataset = DummySeqRecDataset(seq_len=2, num_negatives=2, item_size=item_size)
    return SLR2RecSeqRecTrainer(
        model=model,
        args=args,
        data_collator=DummySeqRecCollator(),
        train_dataset=dataset,
        eval_dataset=dataset,
    )


def test_sl_r2rec_gamma_weights_favor_tail_items(tmp_path: Path) -> None:
    torch.manual_seed(0)
    trainer = _build_trainer(tmp_path, item_size=12)
    tail_and_head = torch.tensor([1, 9], dtype=torch.long)

    gamma = trainer._compute_gamma_weights(tail_and_head)

    expected_tail = torch.exp(torch.tensor(trainer.args.r2rec_alpha_b / trainer.args.r2rec_tau))
    expected_head = torch.exp(torch.tensor(trainer.args.r2rec_alpha_p / trainer.args.r2rec_tau))
    expected = torch.stack([expected_tail, expected_head]) / (expected_tail + expected_head)

    torch.testing.assert_close(gamma, expected)


def test_sl_r2rec_compute_rec_loss_applies_gamma(tmp_path: Path) -> None:
    torch.manual_seed(0)
    trainer = _build_trainer(tmp_path, item_size=20)

    labels = torch.tensor([[1, 10]], dtype=torch.long)
    negative_item_ids = torch.tensor([[2, 3]], dtype=torch.long)
    attention_mask = torch.ones((1, 2), dtype=torch.long)

    user_emb = torch.arange(1 * 2 * trainer.model.config.hidden_size, dtype=torch.float32).reshape(
        1,
        2,
        trainer.model.config.hidden_size,
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
    sl_losses = torch.logsumexp(diff_scores / trainer.args.sl_temperature, dim=-1) * trainer.args.sl_temperature

    positive_items_flat = labels.flatten()[mask]
    gamma = trainer._compute_gamma_weights(positive_items_flat)
    expected_loss = (sl_losses * gamma).mean()

    torch.testing.assert_close(loss, expected_loss)


def test_sl_r2rec_compute_rec_loss_normalizes_when_requested(tmp_path: Path) -> None:
    torch.manual_seed(0)
    trainer = _build_trainer(tmp_path, item_size=18)

    labels = torch.tensor([[1, 7]], dtype=torch.long)
    negative_item_ids = torch.tensor([[4, 5]], dtype=torch.long)
    attention_mask = torch.ones((1, 2), dtype=torch.long)

    user_emb = torch.arange(1 * 2 * trainer.model.config.hidden_size, dtype=torch.float32).reshape(
        1,
        2,
        trainer.model.config.hidden_size,
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
    diff_scores = negative_scores_flat - positive_scores_flat.unsqueeze(-1)
    sl_losses = torch.logsumexp(diff_scores / trainer.args.sl_temperature, dim=-1) * trainer.args.sl_temperature

    gamma = trainer._compute_gamma_weights(labels.flatten()[mask])
    expected_loss = (sl_losses * gamma).mean()

    torch.testing.assert_close(loss, expected_loss)
