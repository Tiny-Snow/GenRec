from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import TrainerCallback
from transformers.generation.utils import GenerateBeamEncoderDecoderOutput
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

from genrec.models.model_genrec.base import GenRecModel, GenRecModelConfig, GenRecOutput
from genrec.trainers.trainer_genrec.base import GenRecTrainer, GenRecTrainingArguments


class _DummyEncoder(nn.Module):
    def __init__(self, embedding: nn.Embedding) -> None:
        super().__init__()
        self.embedding = embedding

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **_: Any,
    ) -> BaseModelOutputWithPastAndCrossAttentions:
        hidden_states = inputs_embeds if inputs_embeds is not None else self.embedding(input_ids)
        return BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=hidden_states)

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.embedding = value  # type: ignore[assignment]


class _DummyDecoder(nn.Module):
    def __init__(self, embedding: nn.Embedding) -> None:
        super().__init__()
        self.embedding = embedding

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        **_: Any,
    ) -> BaseModelOutputWithPastAndCrossAttentions:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        elif input_ids is not None:
            hidden_states = self.embedding(input_ids)
        elif encoder_hidden_states is not None:
            hidden_states = encoder_hidden_states
        else:  # pragma: no cover - defensive branch
            hidden_states = torch.zeros(1, 1, self.embedding.embedding_dim)
        return BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=hidden_states)

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.embedding = value  # type: ignore[assignment]


class DummyGenRecDataset(Dataset[Dict[str, torch.Tensor]]):
    def __init__(
        self,
        *,
        length: int = 4,
        history_windows: int = 3,
        sid_width: int = 2,
        item_size: int = 12,
    ) -> None:
        self.length = length
        self.history_windows = history_windows
        self.sid_width = sid_width
        self.item_size = item_size
        self.item_popularity = np.arange(item_size + 1, dtype=np.int64)
        self.train_item_popularity = self.item_popularity
        self._sid_sequences = {
            item_id: np.arange(item_id, item_id + sid_width, dtype=np.int64) for item_id in range(1, item_size + 1)
        }
        self._sid2item = {tuple(seq.tolist()): item_id for item_id, seq in self._sid_sequences.items()}

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        base_item = (idx % self.item_size) + 1
        labels = torch.as_tensor(self._sid_sequences[base_item], dtype=torch.long)
        flat_len = self.history_windows * self.sid_width
        start = base_item * 10
        input_ids = torch.arange(start, start + flat_len, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def get_prefix_allowed_tokens_fn(self):
        def allowed_tokens(_: int, __: torch.Tensor) -> list[int]:
            return list(range(1, self.item_size + 1))

        return allowed_tokens

    @property
    def sid2item(self) -> Dict[Tuple[int, ...], int]:
        return self._sid2item


class DummyGenRecCollator:
    def __call__(self, batch: list[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        return {key: torch.stack([sample[key] for sample in batch], dim=0) for key in batch[0].keys()}


class DummyTrainerCallback(TrainerCallback):
    pass


class DummyGenRecModel(GenRecModel[GenRecModelConfig, GenRecOutput, BaseModelOutputWithPastAndCrossAttentions]):
    config_class = GenRecModelConfig
    output_class = GenRecOutput

    def __init__(self, config: GenRecModelConfig, sid_width: int, model_loss_value: float | None = None) -> None:
        super().__init__(config)
        self.sid_width = sid_width
        self._model_loss_value = model_loss_value
        self._encoder = _DummyEncoder(self.shared)
        self._decoder = _DummyDecoder(self.shared)

    @property
    def encoder(self) -> nn.Module:
        return self._encoder

    @property
    def decoder(self) -> nn.Module:
        return self._decoder

    def forward(self, *args: Any, output_model_loss: bool | None = None, **kwargs: Any) -> GenRecOutput:
        outputs = super().forward(*args, output_model_loss=output_model_loss, **kwargs)
        if output_model_loss and self._model_loss_value is not None and outputs.logits is not None:
            outputs.model_loss = torch.tensor(self._model_loss_value, device=outputs.logits.device)
        return outputs

    def generate(self, input_ids: torch.Tensor, generation_config=None, **kwargs: Any) -> GenerateBeamEncoderDecoderOutput:  # type: ignore[override]
        cfg = generation_config
        num_beams = cfg.num_beams if cfg is not None else 1
        width = cfg.max_new_tokens if cfg is not None else self.sid_width
        batch = input_ids.size(0)
        device = input_ids.device
        sequences = []
        for batch_idx in range(batch):
            for beam_idx in range(num_beams):
                start = batch_idx * num_beams + beam_idx + 1
                seq = torch.arange(start, start + width, dtype=torch.long, device=device)
                sequences.append(seq)
        stacked = torch.stack(sequences, dim=0)
        return GenerateBeamEncoderDecoderOutput(sequences=stacked)


class MinimalGenRecTrainer(GenRecTrainer[DummyGenRecModel, GenRecTrainingArguments]):
    def compute_rec_loss(
        self,
        inputs: Dict[str, torch.Tensor],
        outputs: GenRecOutput,
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return torch.zeros((), device=outputs.logits.device)


class RecLossTrackingGenRecTrainer(GenRecTrainer[DummyGenRecModel, GenRecTrainingArguments]):
    def __init__(self, *args: Any, rec_loss_value: float, **kwargs: Any) -> None:
        self.rec_loss_value = rec_loss_value
        self.last_seen_num_items: torch.Tensor | None = None
        super().__init__(*args, **kwargs)

    def compute_rec_loss(
        self,
        inputs: Dict[str, torch.Tensor],
        outputs: GenRecOutput,
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if num_items_in_batch is not None:
            self.last_seen_num_items = num_items_in_batch.detach().clone()
        return torch.full((), self.rec_loss_value, device=outputs.logits.device)


def build_genrec_training_args(
    base_dir: Path,
    eval_interval: int = 2,
    model_loss_weight: float = 1.0,
    metrics: Tuple[Tuple[str, Dict[str, Any]], ...] = (("hr", {}),),
    top_k: Tuple[int, ...] = (1, 2),
    num_beams: int | None = None,
    *,
    args_cls: type[GenRecTrainingArguments] = GenRecTrainingArguments,
    **extra_kwargs: Any,
) -> GenRecTrainingArguments:
    resolved_beams = max(num_beams or max(top_k), max(top_k))
    output_dir = base_dir / f"genrec_trainer_tests_{args_cls.__name__.lower()}_{eval_interval}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return args_cls(
        output_dir=str(output_dir),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        remove_unused_columns=False,
        eval_strategy="epoch",
        logging_steps=1,
        max_steps=1,
        eval_interval=eval_interval,
        metrics=metrics,
        top_k=top_k,
        num_beams=resolved_beams,
        model_loss_weight=model_loss_weight,
        use_cpu=True,
        **extra_kwargs,
    )
