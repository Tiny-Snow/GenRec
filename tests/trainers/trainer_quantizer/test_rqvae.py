from __future__ import annotations

from typing import Any, Dict

import torch

from genrec.models.model_quantizer.base import QuantizerModel, QuantizerModelConfig, QuantizerOutput
from genrec.trainers.trainer_quantizer.rqvae import RQVAEQuantizerTrainer, RQVAEQuantizerTrainingArguments


class DummyQuantizerModel(QuantizerModel[QuantizerModelConfig, QuantizerOutput]):
    config_class = QuantizerModelConfig

    def __init__(self, config: QuantizerModelConfig) -> None:
        super().__init__(config)

    def forward(
        self,
        item_id: torch.Tensor,
        item_embedding: torch.Tensor,
        output_loss: bool = False,
        output_model_loss: bool = False,
        output_embeddings: bool = False,
        **kwargs: Any,
    ) -> QuantizerOutput:
        return QuantizerOutput(semantic_ids=torch.zeros(item_embedding.shape[0], 1, dtype=torch.long))

    def initialize_codebooks(self, item_embeddings: torch.Tensor, **kwargs: Any) -> None:
        return None


class DummyDataset:
    def __len__(self) -> int:
        return 2

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "item_id": torch.tensor(idx + 1, dtype=torch.long),
            "item_embedding": torch.randn(4),
        }


class DummyCollator:
    def __call__(self, batch: list[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        item_id = torch.stack([item["item_id"] for item in batch])
        item_embedding = torch.stack([item["item_embedding"] for item in batch])
        return {"item_id": item_id, "item_embedding": item_embedding}


class MinimalRQVAETrainer(RQVAEQuantizerTrainer[DummyQuantizerModel]):
    def initialize_codebooks(self) -> None:
        return None


def test_rqvae_trainer_compute_quantizer_loss(tmp_path) -> None:
    args = RQVAEQuantizerTrainingArguments(
        output_dir=str(tmp_path),
        codebook_loss_weight=0.5,
        commitment_loss_weight=0.25,
    )
    model = DummyQuantizerModel(
        QuantizerModelConfig(
            embed_dim=4,
            hidden_sizes=(3,),
            num_codebooks=1,
            codebook_size=8,
            codebook_dim=2,
        )
    )

    trainer = MinimalRQVAETrainer(
        model=model,
        args=args,
        data_collator=DummyCollator(),
        train_dataset=DummyDataset(),
        eval_dataset=None,
        compute_metrics=lambda _: {},
    )

    outputs = QuantizerOutput(
        semantic_ids=torch.zeros(2, 1, dtype=torch.long),
        reconstruction_loss=torch.tensor([1.0, 2.0]),
        codebook_loss=torch.tensor([0.5, 1.5]),
        commitment_loss=torch.tensor([2.0, 0.0]),
    )
    loss = trainer.compute_quantizer_loss(inputs={}, outputs=outputs)

    expected = (
        outputs.reconstruction_loss.mean()
        + args.codebook_loss_weight * outputs.codebook_loss.mean()
        + args.commitment_loss_weight * outputs.commitment_loss.mean()
    )
    torch.testing.assert_close(loss, expected)
