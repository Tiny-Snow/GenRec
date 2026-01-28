from __future__ import annotations

from functools import partial
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd
import torch

from genrec.datasets import DatasetSplitLiteral, QuantizerCollator, QuantizerDataset
from genrec.models.model_quantizer.base import QuantizerModel, QuantizerModelConfig, QuantizerOutput
from genrec.trainers.trainer_quantizer.base import QuantizerTrainer, QuantizerTrainingArguments
from genrec.trainers.trainer_quantizer.utils.callbacks import EpochIntervalEvalCallback, HardStopCallback
from genrec.trainers.trainer_quantizer.utils.evaluations import compute_quantizer_metrics
from transformers import TrainerCallback


class DummyEncoder:
    embedding_dim: int = 4

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        base = np.arange(len(texts) * self.embedding_dim, dtype=np.float32)
        return base.reshape(len(texts), self.embedding_dim)


class DummyQuantizerModel(QuantizerModel[QuantizerModelConfig, QuantizerOutput]):
    config_class = QuantizerModelConfig

    def __init__(self, config: QuantizerModelConfig) -> None:
        super().__init__(config)
        self.initialize_called = False
        self.last_init_embeddings: torch.Tensor | None = None
        self._dummy_param = torch.nn.Parameter(torch.zeros(1))

    def forward(
        self,
        item_id: torch.Tensor,
        item_embedding: torch.Tensor,
        output_loss: bool = False,
        output_model_loss: bool = False,
        output_embeddings: bool = False,
        **kwargs: Any,
    ) -> QuantizerOutput:
        batch_size = item_embedding.shape[0]
        semantic_ids = torch.zeros(batch_size, self.config.num_codebooks, dtype=torch.long)
        reconstruction_loss = torch.full((batch_size,), 1.0, device=item_embedding.device) if output_loss else None
        codebook_loss = torch.full((batch_size,), 2.0, device=item_embedding.device) if output_loss else None
        commitment_loss = torch.full((batch_size,), 3.0, device=item_embedding.device) if output_loss else None
        model_loss = torch.tensor(4.0, device=item_embedding.device) if output_model_loss else None
        return QuantizerOutput(
            semantic_ids=semantic_ids,
            reconstruction_loss=reconstruction_loss,
            codebook_loss=codebook_loss,
            commitment_loss=commitment_loss,
            model_loss=model_loss,
        )

    def initialize_codebooks(self, item_embeddings: torch.Tensor, **kwargs: Any) -> None:
        self.initialize_called = True
        self.last_init_embeddings = item_embeddings


class MinimalQuantizerTrainer(QuantizerTrainer[DummyQuantizerModel, QuantizerTrainingArguments]):
    def compute_quantizer_loss(
        self,
        inputs: dict[str, torch.Tensor],
        outputs: QuantizerOutput,
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert outputs.reconstruction_loss is not None
        return outputs.reconstruction_loss.mean()


def _make_interaction_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "UserID": np.array([0, 1], dtype=np.int64),
            "ItemID": [
                [1, 2, 3],
                [2, 3, 4],
            ],
            "Timestamp": [
                [1, 2, 3],
                [1, 2, 3],
            ],
        }
    )


def _make_textual_frame(item_pool: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ItemID": np.arange(1, item_pool + 1, dtype=np.int64),
            "Title": [f"Item {idx}" for idx in range(1, item_pool + 1)],
        }
    )


def _build_quantizer_dataset() -> QuantizerDataset:
    interaction_frame = _make_interaction_frame()
    textual_frame = _make_textual_frame(item_pool=4)
    return QuantizerDataset(
        interaction_data_path=interaction_frame,
        max_seq_length=4,
        min_seq_length=1,
        textual_data_path=textual_frame,
        lm_encoder=DummyEncoder(),
    )


def test_quantizer_trainer_defaults_and_initialize_codebooks(tmp_path) -> None:
    dataset = _build_quantizer_dataset()
    collator = QuantizerCollator(dataset)
    model = DummyQuantizerModel(
        QuantizerModelConfig(
            embed_dim=4,
            hidden_sizes=(3,),
            num_codebooks=2,
            codebook_size=8,
            codebook_dim=2,
        )
    )
    args = QuantizerTrainingArguments(
        output_dir=str(tmp_path),
        eval_interval=2,
        train_stop_epoch=5,
    )

    trainer = MinimalQuantizerTrainer(
        model=model,
        args=args,
        data_collator=collator,
        train_dataset=dataset,
    )

    assert trainer.label_names == ["item_id"]
    assert isinstance(trainer.compute_metrics, partial)
    assert trainer.compute_metrics.func is compute_quantizer_metrics
    assert trainer.compute_metrics.keywords["metrics"] == args.metrics
    assert trainer.compute_metrics.keywords["codebook_size"] == model.config.codebook_size
    assert trainer.compute_metrics.keywords["train_dataset"] is dataset

    callback_types = tuple(type(cb) for cb in trainer.callback_handler.callbacks)
    assert any(isinstance(cb, EpochIntervalEvalCallback) for cb in trainer.callback_handler.callbacks), callback_types
    assert any(isinstance(cb, HardStopCallback) for cb in trainer.callback_handler.callbacks), callback_types

    assert model.initialize_called is True
    assert model.last_init_embeddings is not None
    torch.testing.assert_close(model.last_init_embeddings.cpu(), torch.from_numpy(dataset.item_textual_embeddings))


def test_quantizer_trainer_compute_loss_with_model_loss(tmp_path) -> None:
    dataset = _build_quantizer_dataset()
    collator = QuantizerCollator(dataset)
    model = DummyQuantizerModel(
        QuantizerModelConfig(
            embed_dim=4,
            hidden_sizes=(3,),
            num_codebooks=1,
            codebook_size=4,
            codebook_dim=2,
        )
    )
    args = QuantizerTrainingArguments(
        output_dir=str(tmp_path),
        model_loss_weight=0.5,
    )

    trainer = MinimalQuantizerTrainer(
        model=model,
        args=args,
        data_collator=collator,
        train_dataset=dataset,
    )

    model.train()
    batch = [dataset[0], dataset[1]]
    inputs = collator(batch)

    loss, output_dict = trainer.compute_loss(trainer.model, inputs, return_outputs=True)

    expected = torch.tensor(1.0 + 4.0 * args.model_loss_weight)
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(output_dict["loss"], expected)
    assert output_dict["semantic_ids"].shape == (2, model.config.num_codebooks)
    assert output_dict["item_id"].shape == (2,)


def test_quantizer_trainer_compute_loss_without_model_loss(tmp_path) -> None:
    dataset = _build_quantizer_dataset()
    collator = QuantizerCollator(dataset)
    model = DummyQuantizerModel(
        QuantizerModelConfig(
            embed_dim=4,
            hidden_sizes=(3,),
            num_codebooks=1,
            codebook_size=4,
            codebook_dim=2,
        )
    )
    args = QuantizerTrainingArguments(output_dir=str(tmp_path))

    trainer = MinimalQuantizerTrainer(
        model=model,
        args=args,
        data_collator=collator,
        train_dataset=dataset,
    )

    model.eval()
    batch = [dataset[0]]
    inputs = collator(batch)

    loss = trainer.compute_loss(trainer.model, inputs, return_outputs=False)
    torch.testing.assert_close(loss, torch.tensor(1.0))


def test_quantizer_trainer_respects_custom_callbacks(tmp_path) -> None:
    dataset = _build_quantizer_dataset()
    collator = QuantizerCollator(dataset)
    model = DummyQuantizerModel(
        QuantizerModelConfig(
            embed_dim=4,
            hidden_sizes=(3,),
            num_codebooks=1,
            codebook_size=4,
            codebook_dim=2,
        )
    )
    args = QuantizerTrainingArguments(output_dir=str(tmp_path))

    class DummyCallback(TrainerCallback):
        pass

    custom_callback = DummyCallback()

    def custom_metrics(_):
        return {"custom": 0.0}

    trainer = MinimalQuantizerTrainer(
        model=model,
        args=args,
        data_collator=collator,
        train_dataset=dataset,
        compute_metrics=custom_metrics,
        callbacks=[custom_callback],
    )

    assert trainer.compute_metrics is custom_metrics
    assert any(isinstance(cb, DummyCallback) for cb in trainer.callback_handler.callbacks)
