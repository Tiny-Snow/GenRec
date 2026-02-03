"""Binary Cross-Entropy (BCE) Trainer for Sequential Recommendation Tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, TypeVar, Union, Tuple, Dict

from jaxtyping import Bool, Float, Int
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...models import SeqRecModel, SeqRecOutput
from .base import SeqRecTrainer, SeqRecTrainerFactory, SeqRecTrainingArguments, SeqRecTrainingArgumentsFactory

__all__ = [
    "BCEMOJITOSeqRecTrainingArguments",
    "BCEMOJITOSeqRecTrainer",
]


_SeqRecModel = TypeVar("_SeqRecModel", bound="SeqRecModel[Any, Any]")


@SeqRecTrainingArgumentsFactory.register("bce_mojito")
@dataclass
class BCEMOJITOSeqRecTrainingArguments(SeqRecTrainingArguments):
    """Training arguments for `BCEMOJITOSeqRecTrainer`."""

    global_loss_weight: float = field(
        default=0.0,
        metadata={"help": "Weight for the global loss term. Default is 0.0."},
    )


@SeqRecTrainerFactory.register("bce_mojito")
class BCEMOJITOSeqRecTrainer(SeqRecTrainer[_SeqRecModel, BCEMOJITOSeqRecTrainingArguments]):
    """Binary Cross-Entropy (BCE) Trainer for Sequential Recommendation Tasks.

    This trainer extends the base `SeqRecTrainer` to implement the binary cross-entropy
    loss function, which is commonly used in sequential recommendation tasks. No additional
    training arguments are required beyond those provided by the base class.
    """

    args: BCEMOJITOSeqRecTrainingArguments
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

        assert inputs["negative_item_ids"] is not None, "Negative item IDs must be provided for BCE loss."
        negative_items: Int[torch.Tensor, "B N"] = inputs["negative_item_ids"]
        negative_emb: Float[torch.Tensor, "B N d"] = self.model.embed_tokens(negative_items)
        if norm_embeddings:
            negative_emb = F.normalize(negative_emb, p=2, dim=-1)

        # get positive and negative scores by dot product with output user embeddings
        user_emb: Float[torch.Tensor, "B L d"] = outputs.last_hidden_state
        if norm_embeddings:
            user_emb = F.normalize(user_emb, p=2, dim=-1)
        positive_scores: Float[torch.Tensor, "B L"] = torch.sum(user_emb * positive_emb, dim=-1)
        negative_scores: Float[torch.Tensor, "B L N"] = torch.matmul(user_emb, negative_emb.transpose(-1, -2))

        # flatten scores and mask out padding positions
        attention_mask: Bool[torch.Tensor, "B*L"]
        attention_mask = inputs["attention_mask"].bool().flatten()

        positive_scores_flat: Float[torch.Tensor, "M"]
        positive_scores_flat = positive_scores.flatten()[attention_mask]

        negative_scores_flat: Float[torch.Tensor, "M N"]
        negative_scores_flat = negative_scores.reshape(-1, negative_scores.size(-1))[attention_mask]

        # calculate BCE loss
        positive_loss: Float[torch.Tensor, ""] = -F.logsigmoid(positive_scores_flat).mean()
        negative_loss: Float[torch.Tensor, ""] = -F.logsigmoid(-negative_scores_flat).mean()

        local_loss: Float[torch.Tensor, ""] = positive_loss + negative_loss

        if hasattr(outputs, "adaptive_last_hidden_state"):
            global_user_emb = outputs.adaptive_last_hidden_state
        else:
            global_user_emb = self.model.compute_global_user_embedding(inputs, outputs)

        global_pos_scores = torch.sum(global_user_emb * positive_emb, dim=-1)
        global_neg_scores = torch.matmul(global_user_emb, negative_emb.transpose(-1, -2))

        global_pos_scores_flat = global_pos_scores.flatten()[attention_mask]
        global_neg_scores_flat = global_neg_scores.reshape(-1, global_neg_scores.size(-1))[attention_mask]
        global_loss = -F.logsigmoid(global_pos_scores_flat).mean() - F.logsigmoid(-global_neg_scores_flat).mean()

        global_loss_weight: float = getattr(self.args, "global_loss_weight", 0.1)
        loss = local_loss + global_loss_weight * global_loss

        return loss
