"""Softmax Loss (SL) + ReSN Trainer for Sequential Recommendation Tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, TypeVar, Union

from jaxtyping import Bool, Float, Int
import torch
import torch.nn.functional as F

from ...models import SeqRecModel, SeqRecOutput
from .base import SeqRecTrainer, SeqRecTrainerFactory, SeqRecTrainingArguments, SeqRecTrainingArgumentsFactory

__all__ = [
    "SLReSNSeqRecTrainingArguments",
    "SLReSNSeqRecTrainer",
]


_SeqRecModel = TypeVar("_SeqRecModel", bound="SeqRecModel[Any, Any]")


@SeqRecTrainingArgumentsFactory.register("sl_resn")
@dataclass
class SLReSNSeqRecTrainingArguments(SeqRecTrainingArguments):
    """Training arguments for `SLReSNSeqRecTrainer`."""

    sl_temperature: float = field(
        default=0.05,
        metadata={"help": "Temperature parameter for Softmax Loss. Default is 0.05."},
    )

    resn_weight: float = field(
        default=0.0001,
        metadata={"help": "Weight for the ReSN loss component, i.e., β in the ReSN paper. Default is 0.0001."},
    )


@SeqRecTrainerFactory.register("sl_resn")
class SLReSNSeqRecTrainer(SeqRecTrainer[_SeqRecModel, SLReSNSeqRecTrainingArguments]):
    """Softmax Loss (SL) + ReSN Trainer for Sequential Recommendation Tasks.

    This trainer extends the base `SeqRecTrainer` to implement the softmax loss function with the
    ReSN regularization term, which is a SOTA spectral-norm-based popularity debiasing method in
    collaborative filtering.

    .. note::
        Here, we adapt it for sequential recommendation tasks --- the average representation of
        the model outputs in each step is treated as user embeddings.

    References:
        - How Do Recommendation Models Amplify Popularity Bias? An Analysis from the Spectral Perspective. WSDM '25.
    """

    args: SLReSNSeqRecTrainingArguments
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
        negative_scores: Float[torch.Tensor, "B L N"] = torch.matmul(user_emb, negative_emb.transpose(-1, -2))

        # flatten scores and mask out padding positions
        attention_mask_flat: Bool[torch.Tensor, "B*L"]
        attention_mask_flat = inputs["attention_mask"].bool().flatten()

        positive_scores_flat: Float[torch.Tensor, "M"]
        positive_scores_flat = positive_scores.flatten()[attention_mask_flat]

        negative_scores_flat: Float[torch.Tensor, "M N"]
        negative_scores_flat = negative_scores.reshape(-1, negative_scores.size(-1))[attention_mask_flat]

        # calculate SL loss
        diff_scores: Float[torch.Tensor, "M N"] = negative_scores_flat - positive_scores_flat.unsqueeze(-1)
        tau = self.args.sl_temperature
        sl_loss: Float[torch.Tensor, ""] = (torch.logsumexp(diff_scores / tau, dim=-1) * tau).mean()

        # get average user embedding for each sequence in the batch
        # NOTE: average over all positions including padding leads to better stability
        attention_mask: Bool[torch.Tensor, "B L"] = inputs["attention_mask"].bool()
        user_emb_sum: Float[torch.Tensor, "B d"] = torch.sum(user_emb * attention_mask.unsqueeze(-1), dim=1)
        user_emb_avg: Float[torch.Tensor, "B d"] = user_emb_sum / attention_mask.size(1)

        # compute ReSN regularization loss
        item_emb: Float[torch.Tensor, "I d"] = self.model.item_embed_weight[1:]  # exclude padding item
        if norm_embeddings:
            item_emb = F.normalize(item_emb, p=2, dim=-1)
        e: Float[torch.Tensor, "B 1"] = torch.ones(user_emb_avg.size(0), 1, device=user_emb_avg.device)
        V_UT_e: Float[torch.Tensor, "1 I"] = e.T @ user_emb_avg @ item_emb.T
        U_VT_V_UT_e: Float[torch.Tensor, "1 B"] = V_UT_e @ item_emb @ user_emb_avg.T

        r_norm: Float[torch.Tensor, ""] = V_UT_e.reshape(-1).norm(p=2)
        Yr_norm: Float[torch.Tensor, ""] = U_VT_V_UT_e.reshape(-1).norm(p=2)
        resn_loss: Float[torch.Tensor, ""] = Yr_norm**2 / (r_norm**2 + 1e-8)

        # combine losses
        loss: Float[torch.Tensor, ""] = sl_loss + self.args.resn_weight * resn_loss

        return loss
