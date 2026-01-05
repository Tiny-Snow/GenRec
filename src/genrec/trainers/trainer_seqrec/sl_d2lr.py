"""Softmax Loss (SL) + D2LR Trainer for Sequential Recommendation Tasks."""

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
    "SLD2LRSeqRecTrainer",
    "SLD2LRSeqRecTrainingArguments",
]


_SeqRecModel = TypeVar("_SeqRecModel", bound="SeqRecModel[Any, Any]")


@SeqRecTrainingArgumentsFactory.register("sl_d2lr")
@dataclass
class SLD2LRSeqRecTrainingArguments(SeqRecTrainingArguments):
    """Training arguments for `SLD2LRSeqRecTrainer`."""

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

    d2lr_ips_temperature: float = field(
        default=0.1,
        metadata={"help": "Temperature parameter for IPS weighting in D2LR. Default is 0.1."},
    )


@SeqRecTrainerFactory.register("sl_d2lr")
class SLD2LRSeqRecTrainer(SeqRecTrainer[_SeqRecModel, SLD2LRSeqRecTrainingArguments]):
    """D2LR Trainer for Sequential Recommendation Tasks.

    This trainer extends the base `SeqRecTrainer` to implement the D2LR loss function, an
    IPS-based popularity debiasing method in generative recommendation. In sequential
    recommendation, it is essentially equivalent to applying IPS weighting to the item-wise
    Softmax Loss. To stabilize training, we apply a temperature parameter to the IPS weights.

    References:
        - Dual Debiasing in LLM-based Recommendation. SIGIR '25.
    """

    args: SLD2LRSeqRecTrainingArguments
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
        attention_mask_flat: Bool[torch.Tensor, "B*L"]
        attention_mask_flat = inputs["attention_mask"].bool().flatten()

        positive_scores_flat: Float[torch.Tensor, "M"]
        positive_scores_flat = positive_scores.flatten()[attention_mask_flat]

        negative_scores_flat: Float[torch.Tensor, "M N"]
        negative_scores_flat = negative_scores.reshape(-1, negative_scores.size(-1))[attention_mask_flat]

        # calculate SL loss
        all_scores: Float[torch.Tensor, "M N+1"]
        all_scores = torch.cat([positive_scores_flat.unsqueeze(-1), negative_scores_flat], dim=-1)
        tau = self.args.sl_temperature
        pos_log_probs: Float[torch.Tensor, "M"] = F.log_softmax(all_scores / tau, dim=-1)[:, 0]
        sl_losses: Float[torch.Tensor, "M"] = -pos_log_probs

        # calculate IPS weights based on item popularity
        self.train_dataset: SeqRecDataset
        item_popularity: Float[torch.Tensor, "I+1"]
        item_popularity = torch.as_tensor(
            self.train_dataset.train_item_popularity,
            device=positive_items.device,
            dtype=torch.float32,
        )
        all_popularity = item_popularity[1:].sum()  # exclude padding item (index 0)

        popularity_positive: Float[torch.Tensor, "M"]
        popularity_positive = item_popularity[positive_items.flatten()[attention_mask_flat]]
        popularity_positive = popularity_positive.clamp(min=1.0)

        ips: Float[torch.Tensor, "M"]
        ips = (all_popularity / popularity_positive).pow(self.args.d2lr_ips_temperature)
        ips = ips / ips.mean()  # normalize to ensure gradient stability

        loss: Float[torch.Tensor, ""] = (sl_losses * ips).mean()

        return loss
