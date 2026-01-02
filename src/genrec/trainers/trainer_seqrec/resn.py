"""ReSN Trainer for Sequential Recommendation Tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, TypeVar, Union

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int

from ...models import SeqRecModel, SeqRecOutput
from .base import SeqRecTrainer, SeqRecTrainerFactory, SeqRecTrainingArguments, SeqRecTrainingArgumentsFactory

__all__ = [
    "ReSNSeqRecTrainingArguments",
    "ReSNSeqRecTrainer",
]


_SeqRecModel = TypeVar("_SeqRecModel", bound="SeqRecModel[Any, Any]")


@SeqRecTrainingArgumentsFactory.register("resn")
@dataclass
class ReSNSeqRecTrainingArguments(SeqRecTrainingArguments):
    """Training arguments for `ReSNSeqRecTrainer`."""

    resn_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for the ReSN loss component, i.e., β in the ReSN paper."},
    )


@SeqRecTrainerFactory.register("resn")
class ReSNSeqRecTrainer(SeqRecTrainer[_SeqRecModel, ReSNSeqRecTrainingArguments]):
    """ReSN Trainer for Sequential Recommendation Tasks.

    This trainer extends the base `SeqRecTrainer` to implement the ReSN regularization
    loss function, a SOTA spectral-norm-based popularity debiasing method in collaborative
    filtering. No additional training arguments are required beyond those provided by the
    base class.

    .. note::
        Here, we adapt it for sequential recommendation tasks --- the average representation of
        the model outputs in each step is treated as user embeddings.

    References:
        - How Do Recommendation Models Amplify Popularity Bias? An Analysis from the Spectral Perspective. WSDM '25.
    """

    args: ReSNSeqRecTrainingArguments
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

        # compute ReSN regularization loss
        item_emb: Float[torch.Tensor, "I d"] = self.model.item_embed.weight[1:]  # exclude padding item
        e: Float[torch.Tensor, "B 1"] = torch.ones(user_emb_avg.size(0), 1, device=user_emb_avg.device)
        V_UT_e: Float[torch.Tensor, "1 I"] = e.T @ user_emb_avg @ item_emb.T
        U_VT_V_UT_e: Float[torch.Tensor, "1 B"] = V_UT_e @ item_emb @ user_emb_avg.T

        r_norm: Float[torch.Tensor, ""] = V_UT_e.reshape(-1).norm(p=2)
        Yr_norm: Float[torch.Tensor, ""] = U_VT_V_UT_e.reshape(-1).norm(p=2)
        resn_loss: Float[torch.Tensor, ""] = Yr_norm**2 / (r_norm**2 + 1e-8)

        # combine losses
        loss: Float[torch.Tensor, ""] = bce_loss + self.args.resn_weight * resn_loss

        return loss
