from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import TrainerCallback

from genrec.models.model_seqrec.base import SeqRecModel, SeqRecModelConfig, SeqRecOutput
from genrec.trainers.trainer_seqrec.base import SeqRecTrainer, SeqRecTrainingArguments


class DummySeqRecModel(SeqRecModel[SeqRecModelConfig, SeqRecOutput]):
    config_class = SeqRecModelConfig

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs: Any,
    ) -> SeqRecOutput:
        embeddings = self.embed_tokens(input_ids)
        return SeqRecOutput(last_hidden_state=embeddings, model_loss=None)


class DummySeqRecDataset(Dataset):
    def __init__(self, seq_len: int = 3, num_negatives: int = 2, item_size: int = 32) -> None:
        self.seq_len = seq_len
        self.num_negatives = num_negatives
        self.length = 4
        self.item_size = item_size
        self.item_popularity = np.arange(self.item_size + 1, dtype=np.int64)
        self.train_item_popularity = self.item_popularity

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        base = idx + 1
        input_ids = torch.arange(base, base + self.seq_len, dtype=torch.long)
        attention_mask = torch.ones(self.seq_len, dtype=torch.long)
        labels = input_ids.clone()
        negative_item_ids = torch.arange(1, 1 + self.num_negatives, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "negative_item_ids": negative_item_ids,
        }


class DummySeqRecCollator:
    def __call__(self, batch: list[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        collated: Dict[str, torch.Tensor] = {}
        for key in batch[0].keys():
            collated[key] = torch.stack([sample[key] for sample in batch], dim=0)
        return collated


class DummyTrainerCallback(TrainerCallback):
    pass


class MinimalSeqRecTrainer(SeqRecTrainer[DummySeqRecModel, SeqRecTrainingArguments]):
    def compute_rec_loss(
        self,
        inputs: Dict[str, torch.Tensor],
        outputs: SeqRecOutput,
        num_items_in_batch: torch.Tensor | None = None,
        norm_embeddings: bool = False,
    ) -> torch.Tensor:
        return torch.zeros((), device=outputs.last_hidden_state.device)


class DummySeqRecModelWithOptionalLoss(DummySeqRecModel):
    def __init__(self, config: SeqRecModelConfig, model_loss_value: float | None) -> None:
        super().__init__(config)
        self._model_loss_value = model_loss_value

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs: Any,
    ) -> SeqRecOutput:
        embeddings = self.embed_tokens(input_ids)
        model_loss = None
        if self._model_loss_value is not None:
            model_loss = torch.tensor(self._model_loss_value, device=embeddings.device)
        return SeqRecOutput(last_hidden_state=embeddings, model_loss=model_loss)


class RecLossTrackingSeqRecTrainer(SeqRecTrainer[DummySeqRecModelWithOptionalLoss, SeqRecTrainingArguments]):
    def __init__(self, *args: Any, rec_loss_value: float, **kwargs: Any) -> None:
        self.rec_loss_value = rec_loss_value
        self.last_seen_num_items: torch.Tensor | None = None
        super().__init__(*args, **kwargs)

    def compute_rec_loss(
        self,
        inputs: Dict[str, torch.Tensor],
        outputs: SeqRecOutput,
        num_items_in_batch: torch.Tensor | None = None,
        norm_embeddings: bool = False,
    ) -> torch.Tensor:
        if num_items_in_batch is not None:
            self.last_seen_num_items = num_items_in_batch.clone()
        return torch.full((), self.rec_loss_value, device=outputs.last_hidden_state.device)


def build_training_args(
    base_dir: Path,
    eval_interval: int = 2,
    model_loss_weight: float = 1.0,
    metrics: Tuple[Tuple[str, Dict[str, Any]], ...] = (("hr", {}),),
    top_k: Tuple[int, ...] = (5, 10),
    *,
    args_cls: type[SeqRecTrainingArguments] = SeqRecTrainingArguments,
    **extra_kwargs: Any,
) -> SeqRecTrainingArguments:
    output_dir = base_dir / f"seqrec_trainer_tests_{args_cls.__name__.lower()}_{eval_interval}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return args_cls(
        output_dir=str(output_dir),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        use_cpu=True,
        remove_unused_columns=False,
        eval_strategy="epoch",
        logging_steps=1,
        max_steps=1,
        eval_interval=eval_interval,
        metrics=metrics,
        top_k=top_k,
        model_loss_weight=model_loss_weight,
        **extra_kwargs,
    )
