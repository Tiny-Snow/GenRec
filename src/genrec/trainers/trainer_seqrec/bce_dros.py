"""BCE + DROS Trainer for Sequential Recommendation Tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, TypeVar, Union

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int

from ...datasets.dataset_seqrec import SeqRecDataset
from ...models import SeqRecModel, SeqRecOutput
from .base import SeqRecTrainer, SeqRecTrainerFactory, SeqRecTrainingArguments, SeqRecTrainingArgumentsFactory

__all__ = [
    "BCEDROSSeqRecTrainer",
    "BCEDROSSeqRecTrainingArguments",
]


_SeqRecModel = TypeVar("_SeqRecModel", bound="SeqRecModel[Any, Any]")


@SeqRecTrainingArgumentsFactory.register("bce_dros")
@dataclass
class BCEDROSSeqRecTrainingArguments(SeqRecTrainingArguments):
    """Training arguments for `BCEDROSSeqRecTrainer`."""

    dros_temperature: float = field(
        default=0.5,
        metadata={"help": "Temperature parameter for DROS, i.e., β_0 in the DROS paper. Default is 0.5."},
    )

    dros_weight: float = field(
        default=0.1,
        metadata={"help": "Weight for the DROS loss component, i.e., α in the DROS paper. Default is 0.1."},
    )

    popularity_temperature: float = field(
        default=0.05,
        metadata={"help": "Temperature parameter for popularity in DROS, i.e., γ in the DROS paper. Default is 0.05."},
    )


@SeqRecTrainerFactory.register("bce_dros")
class BCEDROSSeqRecTrainer(SeqRecTrainer[_SeqRecModel, BCEDROSSeqRecTrainingArguments]):
    """DROS Trainer for Sequential Recommendation Tasks.

    This trainer extends the base `SeqRecTrainer` to implement the ReSN regularization
    loss function (Regulation with Spectral Norm), which aims to constrain the spectral
    norm of the predicted user-item interaction matrix to mitigate popularity bias.

    .. note::
        In the original DROS implementation, we correct two critical issues in the loss
        computation: (1) the MSE regularization term is computed directly on the unnormalized
        scores rather than the probabilities after sigmoid; (2) the popularity-based weights are
        incorrectly applied to the MSE loss, while they should be applied to the exponentiated
        MSE loss. We have fixed these issues in this implementation.

    Reference:
        - A Generic Learning Framework for Sequential Recommendation with Distribution Shifts. SIGIR '23.
    """

    args: BCEDROSSeqRecTrainingArguments
    model: _SeqRecModel

    def compute_rec_loss(
        self,
        inputs: dict[str, Union[torch.Tensor, Any]],
        outputs: SeqRecOutput,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ) -> Float[torch.Tensor, ""]:
        """Computes the recommendation loss for a batch of inputs and model outputs.

        Args:
            inputs (dict[str, Union[torch.Tensor, Any]]): Dictionary of input tensors, i.e., the
                `SeqRecCollator` output.
            outputs (SeqRecOutput): Model outputs from the forward pass.
            num_items_in_batch (Optional[torch.Tensor]): Optional tensor indicating the number of
                valid items in each sequence in the batch (excluding padding). Here we do not use it.

        Returns:
            Float[torch.Tensor, ""]: Computed recommendation loss as a scalar tensor.
        """
        # get positive and negative item embeddings
        positive_items: Int[torch.Tensor, "B L"] = inputs["labels"]
        positive_items = torch.clamp(positive_items, min=0)  # pad_label: -100 -> 0
        positive_emb: Float[torch.Tensor, "B L d"] = self.model.embed_tokens(positive_items)

        assert inputs["negative_item_ids"] is not None, "Negative item IDs must be provided for BCE loss."
        negative_items: Int[torch.Tensor, "B N"] = inputs["negative_item_ids"]
        negative_emb: Float[torch.Tensor, "B N d"] = self.model.embed_tokens(negative_items)

        # get positive and negative scores by dot product with output user embeddings
        user_emb: Float[torch.Tensor, "B L d"] = outputs.last_hidden_state
        positive_scores: Float[torch.Tensor, "B L"] = torch.sum(user_emb * positive_emb, dim=-1)
        negative_scores: Float[torch.Tensor, "B L N"] = torch.matmul(user_emb, negative_emb.transpose(-1, -2))

        # flatten scores and mask out padding positions
        attention_mask: Bool[torch.Tensor, "B*L"]
        attention_mask = inputs["attention_mask"].bool().flatten()

        positive_scores_flat: Float[torch.Tensor, "M"]
        positive_scores_flat = positive_scores.flatten()[attention_mask]
        positive_probs: Float[torch.Tensor, "M"] = torch.sigmoid(positive_scores_flat)

        negative_scores_flat: Float[torch.Tensor, "M N"]
        negative_scores_flat = negative_scores.reshape(-1, negative_scores.size(-1))[attention_mask]
        negative_probs: Float[torch.Tensor, "M N"] = torch.sigmoid(negative_scores_flat)

        # calculate BCE loss with the same numerics as BCE trainer
        bce_positive_loss: Float[torch.Tensor, ""] = -F.logsigmoid(positive_scores_flat).mean()
        bce_negative_loss: Float[torch.Tensor, ""] = -F.logsigmoid(-negative_scores_flat).mean()

        bce_loss = bce_positive_loss + bce_negative_loss

        # calculate MSE loss
        mse_positive_losses: Float[torch.Tensor, "M"] = (1.0 - positive_probs) ** 2
        mse_negative_losses: Float[torch.Tensor, "M N"] = (negative_probs) ** 2

        # apply popularity-based weights
        self.train_dataset: SeqRecDataset
        item_popularity: Float[torch.Tensor, "I+1"]
        item_popularity = torch.as_tensor(
            self.train_dataset.train_item_popularity,
            device=positive_items.device,
            dtype=torch.float32,
        )
        all_popularity = item_popularity[1:].sum()  # exclude padding item (index 0)

        popularity_positive: Float[torch.Tensor, "M"]
        popularity_positive = item_popularity[positive_items.flatten()[attention_mask]]
        weight_positive: Float[torch.Tensor, "M"]
        weight_positive = (popularity_positive / all_popularity).pow(self.args.popularity_temperature)

        repeat_negative_items: Int[torch.Tensor, "B L N"]
        repeat_negative_items = negative_items.unsqueeze(1).expand(-1, positive_items.size(1), -1)
        popularity_negative: Float[torch.Tensor, "M N"]
        popularity_negative = item_popularity[
            repeat_negative_items.reshape(-1, repeat_negative_items.size(-1))[attention_mask]
        ]
        weight_negative: Float[torch.Tensor, "M N"]
        weight_negative = (popularity_negative / all_popularity).pow(self.args.popularity_temperature)

        weighted_mse_positive: Float[torch.Tensor, "M"]
        weighted_mse_positive = torch.exp(mse_positive_losses / self.args.dros_temperature) * weight_positive
        dro_loss_positive: Float[torch.Tensor, ""] = weighted_mse_positive.sum() / weight_positive.sum()
        dro_loss_positive = self.args.dros_temperature * dro_loss_positive.log()

        weighted_mse_negative: Float[torch.Tensor, "M N"]
        weighted_mse_negative = torch.exp(mse_negative_losses / self.args.dros_temperature) * weight_negative
        dro_loss_negative: Float[torch.Tensor, ""] = weighted_mse_negative.sum() / weight_negative.sum()
        dro_loss_negative = self.args.dros_temperature * dro_loss_negative.log()

        dro_loss = dro_loss_positive + dro_loss_negative

        return bce_loss + self.args.dros_weight * dro_loss
