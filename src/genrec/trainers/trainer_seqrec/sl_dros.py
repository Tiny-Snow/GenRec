"""Softmax Loss (SL) + DROS Trainer for Sequential Recommendation Tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, TypeVar, Union

from jaxtyping import Bool, Float, Int
import torch
import torch.nn.functional as F

from ...datasets.dataset_seqrec import SeqRecDataset
from ...models import SeqRecModel, SeqRecOutput
from .base import SeqRecTrainer, SeqRecTrainerFactory, SeqRecTrainingArguments, SeqRecTrainingArgumentsFactory

__all__ = [
    "SLDROSSeqRecTrainer",
    "SLDROSSeqRecTrainingArguments",
]


_SeqRecModel = TypeVar("_SeqRecModel", bound="SeqRecModel[Any, Any]")


@SeqRecTrainingArgumentsFactory.register("sl_dros")
@dataclass
class SLDROSSeqRecTrainingArguments(SeqRecTrainingArguments):
    """Training arguments for `SLDROSSeqRecTrainer`."""

    sl_temperature: float = field(
        default=0.05,
        metadata={"help": "Temperature parameter for Softmax Loss. Default is 0.05."},
    )

    stepwise_negative_sampling: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to use step-wise negative sampling for SL. If False, item negatives are shared across "
                "all time steps in a sequence. If True, negative samples are resampled for each time step, "
                "while the number of negative samples at each step remains the same. Default is True."
            )
        },
    )

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


@SeqRecTrainerFactory.register("sl_dros")
class SLDROSSeqRecTrainer(SeqRecTrainer[_SeqRecModel, SLDROSSeqRecTrainingArguments]):
    """DROS Trainer for Sequential Recommendation Tasks.

    This trainer extends the base `SeqRecTrainer` to implement the DROS loss function,
    which aims to enhance robustness against popularity shifts in recommendation systems.

    .. note::
        In the original DROS implementation, we correct two critical issues in the loss
        computation: (1) the MSE regularization term is computed directly on the unnormalized
        scores rather than the probabilities after sigmoid; (2) the popularity-based weights are
        incorrectly applied to the MSE loss, while they should be applied to the exponentiated
        MSE loss. We have fixed these issues in this implementation.

    References:
        - A Generic Learning Framework for Sequential Recommendation with Distribution Shifts. SIGIR '23.
    """

    args: SLDROSSeqRecTrainingArguments
    model: _SeqRecModel

    def compute_rec_loss(
        self,
        inputs: dict[str, Union[torch.Tensor, Any]],
        outputs: SeqRecOutput,
        num_items_in_batch: Optional[torch.Tensor] = None,
        norm_embeddings: bool = False,
    ) -> Float[torch.Tensor, ""]:
        """Computes the recommendation loss for a batch of inputs and model outputs.

        Args:
            inputs (dict[str, Union[torch.Tensor, Any]]): Dictionary of input tensors, i.e., the
                `SeqRecCollator` output.
            outputs (SeqRecOutput): Model outputs from the forward pass.
            num_items_in_batch (Optional[torch.Tensor]): Optional tensor indicating the number of
                valid items in each sequence in the batch (excluding padding). Here we do not use it.
            norm_embeddings (bool): Whether to L2-normalize user and item embeddings. Default is False.

        Returns:
            Float[torch.Tensor, ""]: Computed recommendation loss as a scalar tensor.
        """
        # get positive and negative item embeddings
        positive_items: Int[torch.Tensor, "B L"] = inputs["labels"]
        positive_items = torch.clamp(positive_items, min=0)  # pad_label: -100 -> 0
        positive_emb: Float[torch.Tensor, "B L d"] = self.model.embed_tokens(positive_items)
        if norm_embeddings:
            positive_emb = F.normalize(positive_emb, p=2, dim=-1)

        assert inputs["negative_item_ids"] is not None, "Negative item IDs must be provided for softmax loss."
        negative_items: Int[torch.Tensor, "B N"] = inputs["negative_item_ids"]
        negative_emb: Float[torch.Tensor, "B N d"] = self.model.embed_tokens(negative_items)
        if norm_embeddings:
            negative_emb = F.normalize(negative_emb, p=2, dim=-1)

        # get positive and negative scores by dot product with output user embeddings
        user_emb: Float[torch.Tensor, "B L d"] = outputs.last_hidden_state
        if norm_embeddings:
            user_emb = F.normalize(user_emb, p=2, dim=-1)

        positive_scores: Float[torch.Tensor, "B L"] = torch.sum(user_emb * positive_emb, dim=-1)

        negative_scores: Float[torch.Tensor, "B L N"]
        if self.args.stepwise_negative_sampling:  # resample negatives for each timestep
            B, L = positive_items.size()
            stepwise_negative_items: Int[torch.Tensor, "B L N"]
            stepwise_negative_items = torch.randint(
                low=1,
                high=self.item_size + 1,
                size=(B, L, negative_items.size(-1)),
                device=negative_items.device,
            )  # NOTE: no guarantee of uniqueness or disjoint with positive items
            stepwise_negative_emb: Float[torch.Tensor, "B L N d"] = self.model.embed_tokens(stepwise_negative_items)
            if norm_embeddings:
                stepwise_negative_emb = F.normalize(stepwise_negative_emb, p=2, dim=-1)
            negative_scores: Float[torch.Tensor, "B L N"]
            negative_scores = torch.sum(user_emb.unsqueeze(-2) * stepwise_negative_emb, dim=-1)
            # mask out positive items in negatives by assigning a very small score
            mask_pos_in_neg: Bool[torch.Tensor, "B L N"] = stepwise_negative_items == positive_items.unsqueeze(-1)
            negative_scores = negative_scores.masked_fill(mask_pos_in_neg, -5e4)
        else:  # share negatives across all timesteps (i.e., sequence-wise negatives)
            negative_scores: Float[torch.Tensor, "B L N"] = torch.matmul(user_emb, negative_emb.transpose(-1, -2))

        # flatten scores and mask out padding positions
        # NOTE: when norm_embeddings=True, sigmoid may oversmooth scores to be very close to 0.5?
        attention_mask: Bool[torch.Tensor, "B*L"]
        attention_mask = inputs["attention_mask"].bool().flatten()

        positive_scores_flat: Float[torch.Tensor, "M"]
        positive_scores_flat = positive_scores.flatten()[attention_mask]
        positive_probs: Float[torch.Tensor, "M"] = torch.sigmoid(positive_scores_flat)

        negative_scores_flat: Float[torch.Tensor, "M N"]
        negative_scores_flat = negative_scores.reshape(-1, negative_scores.size(-1))[attention_mask]
        negative_probs: Float[torch.Tensor, "M N"] = torch.sigmoid(negative_scores_flat)

        # calculate SL loss
        all_scores: Float[torch.Tensor, "M N+1"]
        all_scores = torch.cat([positive_scores_flat.unsqueeze(-1), negative_scores_flat], dim=-1)
        tau = self.args.sl_temperature
        pos_log_probs: Float[torch.Tensor, "M"] = F.log_softmax(all_scores / tau, dim=-1)[:, 0]
        sl_loss: Float[torch.Tensor, ""] = -pos_log_probs.mean()

        # calculate MSE loss
        mse_positive_losses: Float[torch.Tensor, "M"] = (1.0 - positive_probs) ** 2
        mse_negative_losses: Float[torch.Tensor, "M N"] = negative_probs**2

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

        # compute DRO loss components
        weighted_mse_positive: Float[torch.Tensor, "M"]
        weighted_mse_positive = torch.exp(mse_positive_losses / self.args.dros_temperature) * weight_positive
        dro_loss_positive: Float[torch.Tensor, ""] = weighted_mse_positive.sum() / weight_positive.sum()
        dro_loss_positive = self.args.dros_temperature * dro_loss_positive.log()

        weighted_mse_negative: Float[torch.Tensor, "M N"]
        weighted_mse_negative = torch.exp(mse_negative_losses / self.args.dros_temperature) * weight_negative
        dro_loss_negative: Float[torch.Tensor, ""] = weighted_mse_negative.sum() / weight_negative.sum()
        dro_loss_negative = self.args.dros_temperature * dro_loss_negative.log()

        dro_loss: Float[torch.Tensor, ""] = dro_loss_positive + dro_loss_negative

        loss: Float[torch.Tensor, ""] = sl_loss + self.args.dros_weight * dro_loss

        return loss
