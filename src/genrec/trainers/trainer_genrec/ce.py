"""Cross-Entropy (CE) Trainer for Generative Recommendation Tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, TypeVar, Union

from jaxtyping import Bool, Float, Int
import torch
import torch.nn.functional as F

from ...models import GenRecModel, GenRecOutput
from .base import GenRecTrainer, GenRecTrainerFactory, GenRecTrainingArguments, GenRecTrainingArgumentsFactory

__all__ = [
    "CEGenRecTrainingArguments",
    "CEGenRecTrainer",
]

_GenRecModel = TypeVar("_GenRecModel", bound="GenRecModel[Any, Any, Any]")


@GenRecTrainingArgumentsFactory.register("ce")
@dataclass
class CEGenRecTrainingArguments(GenRecTrainingArguments):
    """Training arguments for `CEGenRecTrainer`."""

    ce_temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature parameter for Cross-Entropy Loss. Default is 1.0."},
    )


@GenRecTrainerFactory.register("ce")
class CEGenRecTrainer(GenRecTrainer[_GenRecModel, CEGenRecTrainingArguments]):
    """Cross-Entropy (CE) Trainer for Generative Recommendation Tasks.

    This trainer extends the base `GenRecTrainer` to implement the cross-entropy
    loss function, which is commonly used in generative recommendation tasks.
    """

    args: CEGenRecTrainingArguments
    model: _GenRecModel

    def compute_rec_loss(
        self,
        inputs: dict[str, Union[torch.Tensor, Any]],
        outputs: GenRecOutput,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ) -> Float[torch.Tensor, ""]:
        """Computes the recommendation loss for a batch of inputs and model outputs.

        Args:
            inputs (dict[str, Union[torch.Tensor, Any]]): Dictionary of input tensors, i.e., the
                `GenRecCollator` output.
            outputs (GenRecOutput): Model outputs from the forward pass.
            num_items_in_batch (Optional[torch.Tensor]): Optional tensor indicating the number of
                valid items in each sequence in the batch (excluding padding).

        Returns:
            Float[torch.Tensor, ""]: Computed recommendation loss as a scalar tensor.
        """
        # get logits, labels
        assert outputs.logits is not None
        logits: Float[torch.Tensor, "B C V"] = outputs.logits
        labels: Int[torch.Tensor, "B C"] = inputs["labels"]

        logits_flat: Float[torch.Tensor, "B*C V"] = logits.view(-1, logits.size(-1))
        labels_flat: Int[torch.Tensor, "B*C"] = labels.view(-1)

        # compute cross-entropy loss
        ce_loss: Float[torch.Tensor, ""] = F.cross_entropy(
            logits_flat / self.args.ce_temperature,
            labels_flat,
            reduction="mean",
        )

        return ce_loss
