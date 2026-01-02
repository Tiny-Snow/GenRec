"""LogDet Trainer for Sequential Recommendation Tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, TypeVar, Union

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int

from ...models import SeqRecModel, SeqRecOutput
from .base import SeqRecTrainer, SeqRecTrainerFactory, SeqRecTrainingArguments, SeqRecTrainingArgumentsFactory

__all__ = [
    "LogDetSeqRecTrainingArguments",
    "LogDetSeqRecTrainer",
]


_SeqRecModel = TypeVar("_SeqRecModel", bound="SeqRecModel[Any, Any]")


@SeqRecTrainingArgumentsFactory.register("logdet")
@dataclass
class LogDetSeqRecTrainingArguments(SeqRecTrainingArguments):
    """Training arguments for `LogDetSeqRecTrainer`."""

    logdet_user_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for the LogDet loss of user embeddings."},
    )

    logdet_item_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for the LogDet loss of item embeddings."},
    )


@SeqRecTrainerFactory.register("logdet")
class LogDetSeqRecTrainer(SeqRecTrainer[_SeqRecModel, LogDetSeqRecTrainingArguments]):
    """LogDet Trainer for Sequential Recommendation Tasks.

    This trainer extends the base `SeqRecTrainer` to implement the LogDet regularization
    loss function, a well-known redundancy reduction method in graph collaborative filtering.

    .. note::
        Here, we adapt it for sequential recommendation tasks --- the average representation of
        the model outputs in each step is treated as user embeddings, and we minimize the LogDet
        of the covariance matrix of user and item embeddings to encourage decorrelation among the
        dimensions. Note that the alignment term in the original LogDet paper, which is originally
        MSE loss, is replaced with the standard BCE loss for recommendation.

    References:
        - Mitigating the Popularity Bias of Graph Collaborative Filtering: A Dimensional Collapse Perspective. NeurIPS '23.

    """

    args: LogDetSeqRecTrainingArguments
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
        attention_mask_flat: Bool[torch.Tensor, "B*L"]
        attention_mask_flat = inputs["attention_mask"].bool().flatten()

        positive_scores_flat: Float[torch.Tensor, "M"]
        positive_scores_flat = positive_scores.flatten()[attention_mask_flat]

        negative_scores_flat: Float[torch.Tensor, "M N"]
        negative_scores_flat = negative_scores.reshape(-1, negative_scores.size(-1))[attention_mask_flat]

        # calculate BCE loss
        positive_bce_loss: Float[torch.Tensor, ""] = -F.logsigmoid(positive_scores_flat).mean()
        negative_bce_loss: Float[torch.Tensor, ""] = -F.logsigmoid(-negative_scores_flat).mean()

        bce_loss = positive_bce_loss + negative_bce_loss

        # get average user embedding for each sequence in the batch
        # NOTE: average over all positions including padding leads to better stability
        attention_mask: Bool[torch.Tensor, "B L"] = inputs["attention_mask"].bool()
        user_emb_sum: Float[torch.Tensor, "B d"] = torch.sum(user_emb * attention_mask.unsqueeze(-1), dim=1)
        user_emb_avg: Float[torch.Tensor, "B d"] = user_emb_sum / attention_mask.size(1)

        # get non-normalized user and item covariance matrix
        user_cov: Float[torch.Tensor, "d d"] = user_emb_avg.T @ user_emb_avg
        item_emb: Float[torch.Tensor, "I d"] = self.model.item_embed.weight[1:]  # exclude padding item
        item_cov: Float[torch.Tensor, "d d"] = item_emb.T @ item_emb

        user_cov = user_cov + 1e-6 * torch.eye(user_cov.size(0), device=user_cov.device)
        item_cov = item_cov + 1e-6 * torch.eye(item_cov.size(0), device=item_cov.device)

        # calculate LogDet regularization loss, add a small identity for numerical stability
        user_logdet_loss = user_cov.trace() - torch.linalg.slogdet(user_cov).logabsdet
        item_logdet_loss = item_cov.trace() - torch.linalg.slogdet(item_cov).logabsdet
        logdet_loss = user_logdet_loss * self.args.logdet_user_weight + item_logdet_loss * self.args.logdet_item_weight

        # combine BCE loss and LogDet loss
        loss: Float[torch.Tensor, ""] = bce_loss + logdet_loss

        return loss
