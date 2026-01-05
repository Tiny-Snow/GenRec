"""Softmax Loss (SL) + R2Rec Trainer for Sequential Recommendation Tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from transformers import EvalPrediction, TrainerCallback

from ...datasets.dataset_seqrec import SeqRecCollator, SeqRecDataset
from ...models import SeqRecModel, SeqRecOutput
from .base import SeqRecTrainer, SeqRecTrainerFactory, SeqRecTrainingArguments, SeqRecTrainingArgumentsFactory

__all__ = [
    "SLR2RecSeqRecTrainingArguments",
    "SLR2RecSeqRecTrainer",
]


_SeqRecModel = TypeVar("_SeqRecModel", bound="SeqRecModel[Any, Any]")


@SeqRecTrainingArgumentsFactory.register("sl_r2rec")
@dataclass
class SLR2RecSeqRecTrainingArguments(SeqRecTrainingArguments):
    """Training arguments for `SLR2RecSeqRecTrainer`."""

    sl_temperature: float = field(
        default=0.05,
        metadata={"help": "Temperature parameter for Softmax Loss. Default is 0.05."},
    )

    r2rec_alpha_b: float = field(
        default=1.0,
        metadata={"help": "Reward factor alpha_b for tail items."},
    )

    r2rec_alpha_p: float = field(
        default=0.0,
        metadata={"help": "Reward factor alpha_p for head items."},
    )

    r2rec_tau: float = field(
        default=0.5,
        metadata={"help": "Temperature parameter for R2Rec reweighting."},
    )

    tail_item_ratio: float = field(
        default=0.8,
        metadata={"help": "Percentage of items considered as tail."},
    )


@SeqRecTrainerFactory.register("sl_r2rec")
class SLR2RecSeqRecTrainer(SeqRecTrainer[_SeqRecModel, SLR2RecSeqRecTrainingArguments]):
    """R2Rec Trainer for Sequential Recommendation Tasks.

    This trainer extends the base `SeqRecTrainer` to implement the R2Rec reweighting method
    while using Softmax Loss as the primary recommendation objective. To stabilize training, a
    temperature parameter is applied to the reweighting factors.

    References:
        - Reembedding and Reweighting are Needed for Tail Item Sequential Recommendation. WWW '25.
    """

    args: SLR2RecSeqRecTrainingArguments
    model: _SeqRecModel

    def __init__(
        self,
        model: _SeqRecModel,
        args: SLR2RecSeqRecTrainingArguments,
        data_collator: SeqRecCollator,
        train_dataset: SeqRecDataset,
        eval_dataset: Optional[SeqRecDataset] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        **kwargs,
    ) -> None:
        """Initializes the SeqRecTrainer with the given model and training arguments.

        Args:
            model (_SeqRecModel): Sequential recommendation model to be trained.
            args (_SeqRecTrainingArguments): Training arguments specific to sequential recommendation.
            data_collator (SeqRecCollator): Data collator that prepares model inputs.
            train_dataset (SeqRecDataset): Dataset used for training.
            eval_dataset (Optional[SeqRecDataset]): Dataset used for evaluation.
            compute_metrics (Optional[Callable[[EvalPrediction], Dict]]): Function used to compute
                metrics during evaluation. Defaults to :func:`compute_seqrec_metrics`.
            callbacks (Optional[List[TrainerCallback]]): Trainer callbacks (defaults to
                `EpochIntervalEvalCallback`).
            **kwargs (Any): Additional keyword arguments forwarded to the base `Trainer`.
        """
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            **kwargs,
        )

        popularity = torch.as_tensor(train_dataset.train_item_popularity, device=model.device, dtype=torch.float32)
        popularity_no_pad = popularity[1:]

        cutoff_index = int(len(popularity_no_pad) * args.tail_item_ratio)
        threshold_pop = torch.topk(popularity_no_pad, k=cutoff_index, largest=False).values.max()

        self.threshold_pop = threshold_pop.item()
        self.is_tail_item_global = popularity <= self.threshold_pop

    def _compute_gamma_weights(
        self,
        item_ids: Int[torch.Tensor, "M"],
    ) -> Float[torch.Tensor, "M"]:
        """Computes the gamma weights for a batch of item IDs based on their tail/head status."""
        is_tail = self.is_tail_item_global[item_ids]
        eta = torch.where(
            is_tail,
            torch.full_like(item_ids, self.args.r2rec_alpha_b, dtype=torch.float32),
            torch.full_like(item_ids, self.args.r2rec_alpha_p, dtype=torch.float32),
        )
        gamma = torch.exp(eta / self.args.r2rec_tau)
        gamma = gamma / gamma.sum()
        return gamma

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
        sl_losses: Float[torch.Tensor, "M"] = torch.logsumexp(diff_scores / tau, dim=-1) * tau

        # reweight loss with R2Rec gamma weights
        positive_item_ids_flat = positive_items.flatten()[attention_mask_flat]
        gamma = self._compute_gamma_weights(positive_item_ids_flat)

        loss: Float[torch.Tensor, ""] = (sl_losses * gamma).mean()

        return loss
