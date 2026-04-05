"""Base trainer for sequential recommendation tasks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, Generic, List, Optional, Sequence, Tuple, Type, TypeVar, Union

from jaxtyping import Float, Int
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import EvalPrediction, Trainer, TrainerCallback, TrainingArguments

from ...datasets import SeqRecCollator, SeqRecDataset
from ...models import SeqRecModel, SeqRecOutput
from ..utils.callbacks import EpochIntervalEvalCallback, HardStopCallback
from ..utils.evaluations import MetricFactory, clip_top_k

__all__ = [
    "compute_seqrec_metrics",
    "SeqRecTrainer",
    "SeqRecTrainerFactory",
    "SeqRecTrainingArguments",
    "SeqRecTrainingArgumentsFactory",
]


_SeqRecModel = TypeVar("_SeqRecModel", bound="SeqRecModel[Any, Any]")
_SeqRecTrainer = TypeVar("_SeqRecTrainer", bound="SeqRecTrainer[Any, Any]")
_SeqRecTrainingArguments = TypeVar("_SeqRecTrainingArguments", bound="SeqRecTrainingArguments")


class SeqRecTrainingArgumentsFactory:  # pragma: no cover - factory class
    """Factory for creating `SeqRecTrainingArguments` instances."""

    _registry: dict[str, Type[SeqRecTrainingArguments]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Type[_SeqRecTrainingArguments]], Type[_SeqRecTrainingArguments]]:
        """Decorator to register a `SeqRecTrainingArguments` implementation."""

        def decorator(
            training_args_cls: Type[_SeqRecTrainingArguments],
        ) -> Type[_SeqRecTrainingArguments]:
            if name in cls._registry:
                raise ValueError(f"SeqRec training arguments '{name}' is already registered.")
            cls._registry[name] = training_args_cls
            return training_args_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs) -> SeqRecTrainingArguments:
        """Creates an instance of a registered `SeqRecTrainingArguments`."""
        if name not in cls._registry:
            raise ValueError(f"SeqRec training arguments '{name}' is not registered.")
        training_args_cls = cls._registry[name]
        return training_args_cls(**kwargs)


class SeqRecTrainerFactory:  # pragma: no cover - factory class
    """Factory for creating `SeqRecTrainer` instances."""

    _registry: dict[str, Type[SeqRecTrainer[Any, Any]]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Type[_SeqRecTrainer]], Type[_SeqRecTrainer]]:
        """Decorator to register a `SeqRecTrainer` implementation."""

        def decorator(trainer_cls: Type[_SeqRecTrainer]) -> Type[_SeqRecTrainer]:
            if name in cls._registry:
                raise ValueError(f"SeqRec trainer '{name}' is already registered.")
            cls._registry[name] = trainer_cls
            return trainer_cls

        return decorator

    @classmethod
    def create(
        cls,
        name: str,
        model: SeqRecModel[Any, Any],
        args: SeqRecTrainingArguments,
        data_collator: SeqRecCollator,
        train_dataset: SeqRecDataset,
        eval_dataset: Optional[SeqRecDataset] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        **kwargs,
    ) -> SeqRecTrainer[Any, Any]:
        """Creates an instance of a registered `SeqRecTrainer`."""
        if name not in cls._registry:
            raise ValueError(f"SeqRec trainer '{name}' is not registered.")
        trainer_cls = cls._registry[name]
        return trainer_cls(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            **kwargs,
        )


@dataclass
class SeqRecTrainingArguments(TrainingArguments):
    """Training arguments for sequential recommendation trainers.

    Args:
        norm_embeddings (bool): Whether to L2-normalize user and item embeddings during loss
            computation and evaluation. If True, both user and item embeddings are normalized to
            unit length, and the dot product is equivalent to cosine similarity. Default is False.
        eval_interval (int): Number of epochs between each evaluation. Default is 5.
        metrics (Sequence[Tuple[str, Dict[str, Any]]]): Metric names and their parameters to
            compute during evaluation. Default is [('hr', {}), ('ndcg', {}), ('popularity', {'p': (0.1, 0.2)}),
            ("unpopularity", {"p": (0.2, 0.4)})].
        model_loss_weight (float): Weight for the model-specific loss when combined with the
            recommendation loss. Default is 0.0.
        top_k (Sequence[int]): Cutoff values for computing top-K metrics during evaluation.
            Default is [1, 5, 10].
    """

    norm_embeddings: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to L2-normalize user and item embeddings during loss computation and evaluation. "
                "If True, both user and item embeddings are normalized to unit length, and the dot product "
                "is equivalent to cosine similarity. Default is False."
            )
        },
    )

    eval_interval: int = field(
        default=5,
        metadata={"help": "Number of epochs between each evaluation. Default is 5."},
    )

    train_stop_epoch: int = field(
        default=-1,
        metadata={"help": "Number of epochs to stop training. Default is -1 (no early stop)."},
    )

    metrics: Sequence[Tuple[str, Dict[str, Any]]] = field(
        default_factory=lambda: [
            ("hr", {}),
            ("ndcg", {}),
            ("popularity", {"p": (0.1, 0.2)}),
            ("unpopularity", {"p": (0.2, 0.4)}),
        ],
        metadata={
            "help": (
                "Metric names and their parameters to compute during evaluation. "
                "Default is [('hr', {}), ('ndcg', {}), ('popularity', {'p': (0.1, 0.2)}), "
                "('unpopularity', {'p': (0.2, 0.4)})]."
            )
        },
    )

    model_loss_weight: float = field(
        default=0.0,
        metadata={
            "help": "Weight for the model-specific loss when combined with the recommendation loss. Default is 0.0."
        },
    )

    top_k: Sequence[int] = field(
        default_factory=lambda: [1, 5, 10],
        metadata={"help": "Cutoff values for computing top-K metrics during evaluation. Default is [1, 5, 10]."},
    )


def compute_seqrec_metrics(
    prediction: EvalPrediction,
    train_dataset: SeqRecDataset,
    top_k: Sequence[int] = (1, 5, 10),
    metrics: Sequence[Tuple[str, Dict[str, Any]]] = (
        ("hr", {}),
        ("ndcg", {}),
        ("popularity", {"p": (0.1, 0.2)}),
        ("unpopularity", {"p": (0.2, 0.4)}),
    ),
    device: Union[torch.device, str, None] = None,
) -> Dict[str, float]:
    """Compute metrics for sequential recommendation tasks.

    Args:
        prediction (EvalPrediction): Object containing model predictions and labels. Predictions are
            expected to be the precomputed top-k item indices per user (shape: ``[num_users, max_k]``).
        train_dataset (SeqRecDataset): Dataset used during training; required for global metrics
            such as popularity-based measurements.
        top_k (Sequence[int]): Cutoff values for computing top-K metrics, determining how many
            predictions to consider for each metric. Default is (1, 5, 10).
        metrics (Sequence[Tuple[str, Dict[str, Any]]]): Metric specifications, where each tuple
            comprises the metric name and an optional parameter dictionary. Default is
            [('hr', {}), ('ndcg', {}), ('popularity', {'p': (0.1, 0.2)}),
            ('unpopularity', {'p': (0.2, 0.4)})].
        device (Union[torch.device, str, None]): Device used for metric computations.
            If None, defaults to CPU. Default is None.

    Returns:
        Dict[str, float]: Dictionary containing computed metric values keyed by metric name.

    .. note::
        As we may call this evaluation function for global metrics (e.g., popularity/fairness),
        you should ensure the `train_dataset` is provided if any global metrics are specified.
        In addition, `batch_eval_metrics` in `SeqRecTrainingArguments` should be set to `False`
        to avoid conflicts.
    """
    torch_device = torch.device(device) if device is not None else torch.device("cpu")

    topk_indices: Int[torch.Tensor, "B K"]
    if isinstance(prediction.predictions, tuple):  # pragma: no cover - rarely used
        topk_indices = torch.as_tensor(prediction.predictions[0], dtype=torch.long, device=torch_device)
        sigma_prop = torch.as_tensor(prediction.predictions[1], dtype=torch.float, device=torch_device)
        sigma = torch.as_tensor(prediction.predictions[2], dtype=torch.float, device=torch_device)
        fro = torch.as_tensor(prediction.predictions[3], dtype=torch.float, device=torch_device)
        pop_attn_score = torch.as_tensor(prediction.predictions[4], dtype=torch.float, device=torch_device)
        unpop_attn_score = torch.as_tensor(prediction.predictions[5], dtype=torch.float, device=torch_device)
        # pop_attn_spearman = torch.as_tensor(prediction.predictions[2], dtype=torch.float, device=torch_device)
        u_v_similarity = torch.as_tensor(prediction.predictions[6], dtype=torch.float, device=torch_device)
    else:
        topk_indices = torch.as_tensor(prediction.predictions, dtype=torch.long, device=torch_device)

    labels: Int[np.ndarray, "B L"]
    if isinstance(prediction.label_ids, tuple):  # pragma: no cover - rarely used
        labels = prediction.label_ids[0].astype(np.int64)
    else:
        labels = prediction.label_ids.astype(np.int64)
    last_step_labels: Int[torch.Tensor, "B"]
    last_step_labels = torch.as_tensor(labels[:, -1], dtype=torch.long, device=torch_device)

    results: Dict[str, float] = {}

    results["sigma_prop"] = sigma_prop.mean().item()
    results["pop_attn_score"] = pop_attn_score.mean().item()
    results["unpop_attn_score"] = unpop_attn_score.mean().item()
    # results["pop_attn_spearman"] = pop_attn_spearman.mean().item()
    results["u_v_similarity"] = u_v_similarity.mean().item()
    results["sigma"] = sigma.mean().item()
    results["fro"] = fro.mean().item()

    for k in top_k:
        sliced_topk_indices = topk_indices[:, :k]
        for metric_name, metric_params in metrics:
            metric_fn = MetricFactory.create(metric_name)
            metric_results = metric_fn(
                topk_indices=sliced_topk_indices,
                labels=last_step_labels,
                train_dataset=train_dataset,
                **metric_params,
            )
            results.update(metric_results)

    return results


class SeqRecTrainer(Trainer, Generic[_SeqRecModel, _SeqRecTrainingArguments], ABC):
    """Base trainer class for sequential recommendation models.

    This class extends the `Trainer` class from the `transformers` library. You should
    implement specific training logic, i.e., `compute_loss`, in subclasses to
    compute the model-agnostic loss for sequential recommendation tasks.

    .. note::
        We set up the default callbacks to include `EpochIntervalEvalCallback`
        which performs evaluation every `eval_interval` epochs (default is 5).

    .. note::
        We also set up the `compute_metrics` function to use `compute_seqrec_metrics` by default.
    """

    args: _SeqRecTrainingArguments
    model: _SeqRecModel

    def __init__(
        self,
        model: _SeqRecModel,
        args: _SeqRecTrainingArguments,
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
            callbacks (Optional[List[TrainerCallback]]): Trainer callbacks. Defaults to
                `[EpochIntervalEvalCallback, HardStopCallback]`.
            **kwargs (Any): Additional keyword arguments forwarded to the base `Trainer`.
        """
        self.item_size = train_dataset.item_size
        self.top_k = tuple(clip_top_k(args.top_k, self.item_size))
        self.max_top_k = max(self.top_k)

        try:
            first_param = next(model.parameters())
            model_device = first_param.device
        except StopIteration:  # pragma: no cover - models without parameters
            model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if compute_metrics is None:
            compute_metrics = partial(
                compute_seqrec_metrics,
                train_dataset=train_dataset,
                top_k=self.top_k,
                metrics=args.metrics,
                device=model_device,
            )

        if callbacks is None:
            callbacks = [
                EpochIntervalEvalCallback(eval_interval=args.eval_interval),
                HardStopCallback(stop_epoch=args.train_stop_epoch),
            ]

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

        # HuggingFace Trainer requires label_names to be set for evaluation, and
        # we set it to ["labels"] by default.
        # Your model's forward method and data collator should ensure that
        # the input batch contains a key "labels" corresponding to the ground truth labels.
        # You may override this attribute if your label key is different in subclasses.
        self.label_names = ["labels"]

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ) -> Union[Float[torch.Tensor, ""], Tuple[Float[torch.Tensor, ""], Dict[str, torch.Tensor]]]:
        """Computes the loss for a batch of inputs. This should be overridden by all subclasses.

        Args:
            model (nn.Module): Model being trained.
            inputs (dict[str, Union[torch.Tensor, Any]]): Dictionary of input tensors, i.e., the
                `SeqRecCollator` output.
            return_outputs (bool): Whether to return the model outputs along with the loss.
            num_items_in_batch (Optional[torch.Tensor]): Optional tensor indicating the number of
                valid items in each sequence in the batch (excluding padding).

        Returns:
            Union[Float[torch.Tensor, ""], Tuple[Float[torch.Tensor, ""], Dict[str, torch.Tensor]]]:
                Either the scalar loss or a tuple containing the loss and a dictionary with
                loss and top-k indices of predicted items when `return_outputs` is True. The loss
                combines the model-specific loss (if provided) with the recommendation loss computed
                via `compute_rec_loss`.
        """
        model = model.module if hasattr(model, "module") else model  # type: ignore - for distributed training
        assert isinstance(model, SeqRecModel), "Model must be an instance of SeqRecModel."

        outputs: SeqRecOutput = model(
            **inputs,
            output_model_loss=model.training,  # only compute model loss during training
            output_attentions=True,
        )
        assert isinstance(outputs, SeqRecOutput), "Model output must be an instance of SeqRecOutput."

        rec_loss = self.compute_rec_loss(inputs, outputs, num_items_in_batch, self.args.norm_embeddings)
        if outputs.model_loss is not None:
            loss = rec_loss + outputs.model_loss * self.args.model_loss_weight
        else:
            loss = rec_loss

        if return_outputs:
            last_step_hidden_states: Float[torch.Tensor, "B d"]
            last_step_hidden_states = outputs.last_hidden_state[:, -1, :]
            item_embed_weight: Float[torch.Tensor, "I+1 d"] = model.item_embed_weight

            if self.args.norm_embeddings:
                last_step_hidden_states = F.normalize(last_step_hidden_states, p=2, dim=-1)
                item_embed_weight = F.normalize(item_embed_weight, p=2, dim=-1)

            logits: Float[torch.Tensor, "B I+1"]
            logits = last_step_hidden_states @ item_embed_weight.T

            S = torch.linalg.svd(logits, full_matrices=False)
            sigma = S[1].max()  # largest singular value, scalar
            fro = torch.linalg.matrix_norm(logits, ord='fro')
            sigma_prop = sigma / fro

            effective_top_k = max(1, min(self.max_top_k, self.item_size))
            _, topk_indices = torch.topk(logits, k=effective_top_k, dim=-1)  # may predict padding index

            output_dict: Dict[str, torch.Tensor] = {
                "loss": loss,
                "topk_indices": topk_indices.detach(),
                "sigma_prop": sigma_prop.detach(),
                "sigma": sigma.detach(),
                "fro": fro.detach(),
            }

            # print("sigma_prop:", sigma_prop.item())
            print("sigma_prop:", sigma_prop.item(), "sigma:", sigma.item(), "fro:", fro.item())

            if outputs.attentions is not None and len(outputs.attentions) > 0:
                last_layer_attn: Float[torch.Tensor, "B H L L"] = outputs.attentions[-1]
                attention_mask: Int[torch.Tensor, "B L"] = inputs["attention_mask"]
                from ...models.modules.utils import create_attention_mask

                causal_mask = create_attention_mask(attention_mask, is_causal=True, mask_value=1).bool()
                last_layer_attn = last_layer_attn.masked_fill(causal_mask, 0.0)

                attn_summed_h: Float[torch.Tensor, "B L L"] = last_layer_attn.sum(dim=1)

                attn_score: Float[torch.Tensor, "B L"] = attn_summed_h.sum(dim=1)

                # U, S, Vh = torch.linalg.svd(attn_summed_h)
                # main_left_vectors: Float[torch.Tensor, "B L"] = U[..., 0]
                # main_right_vectors: Float[torch.Tensor, "B L"] = Vh[..., 0]

                # # normalize
                # # main_left_vectors = main_left_vectors / (main_left_vectors.norm(dim=-1, keepdim=True))
                # # main_right_vectors = main_right_vectors / (main_right_vectors.norm(dim=-1, keepdim=True))

                # cos_similarity_scores: Float[torch.Tensor, "B"] = torch.sum(
                #     main_left_vectors * main_right_vectors, dim=-1
                # )
                # mean_similarity_scores = cos_similarity_scores.mean(dim=0)  # scalar

                # input_popularity = inputs.get("input_popularity")
                # if input_popularity is not None:

                #     # spearman corr between input popularity and attention score, need mask
                #     input_popularity.masked_fill_(attention_mask == 0, 0.0)
                #     input_popularity_flat = input_popularity.view(-1).cpu().numpy()
                #     attn_score_flat = attn_score.view(-1).cpu().numpy()

                #     spearman_corr = np.corrcoef(input_popularity_flat, attn_score_flat)[0, 1]
                #     output_dict["pop_attn_spearman"] = torch.tensor(spearman_corr, device=attn_score.device)
                input_is_pop = inputs.get("input_is_pop")
                if input_is_pop is not None:
                    input_is_pop = input_is_pop.bool()
                    pad_mask = attention_mask == 0

                    pop_mask = input_is_pop & ~pad_mask
                    unpop_mask = ~input_is_pop & ~pad_mask

                    pop_count = pop_mask.sum().float()
                    unpop_count = unpop_mask.sum().float()

                    if pop_count > 0:
                        pop_mean_score = attn_score[pop_mask].sum()
                    else:
                        pop_mean_score = torch.tensor(0.0, device=attn_score.device)

                    if unpop_count > 0:
                        unpop_mean_score = attn_score[unpop_mask].sum()
                    else:
                        unpop_mean_score = torch.tensor(0.0, device=attn_score.device)

                    output_dict["pop_attn_score"] = pop_mean_score.detach()
                    output_dict["unpop_attn_score"] = unpop_mean_score.detach()

                attn_matrix: Float[torch.Tensor, "L L"] = attn_summed_h.mean(
                    dim=0
                )  # average over batch and heads, shape (L, L)
                U, S, Vh = torch.linalg.svd(attn_matrix)
                k = min(10, S.numel())
                left = U[:, :k]  # (L, k)
                right = Vh[:k, :].T  # (L, k)  (transpose rows of Vh)
                # per-component dot and norms
                numer = (left * right).sum(dim=0)  # (k,)
                denom = left.norm(dim=0) * right.norm(dim=0)
                cosines = numer / (denom)  # (k,)
                # simple mean and singular-value-weighted mean
                mean_cos = cosines.mean()
                weights = S[:k] / (S[:k].sum())
                weighted_mean_cos = (cosines * weights).sum()
                # choose weighted_mean_cos as the main metric (more robust to scale)
                mean_similarity_scores = weighted_mean_cos
                output_dict["u_v_similarity"] = mean_similarity_scores.detach()
            return loss, output_dict
        else:
            return loss

    @abstractmethod
    def compute_rec_loss(  # pragma: no cover - abstract method
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
                valid items in each sequence in the batch (excluding padding).
            norm_embeddings (bool): Whether to L2-normalize user and item embeddings. Default is False.

        Returns:
            Float[torch.Tensor, ""]: Computed recommendation loss as a scalar tensor.
        """
        ...
