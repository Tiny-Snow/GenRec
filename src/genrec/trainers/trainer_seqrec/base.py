"""Base trainer for sequential recommendation tasks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, Generic, List, Optional, Sequence, Tuple, Type, TypeVar, Union

import numpy as np
import torch
import torch.nn as nn
from jaxtyping import Float, Int
from transformers import EvalPrediction, Trainer, TrainerCallback, TrainingArguments

from ...datasets import SeqRecCollator, SeqRecDataset
from ...models import SeqRecModel, SeqRecOutput
from ..utils.callbacks import EpochIntervalEvalCallback
from ..utils.evaluations import MetricFactory

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
        eval_interval (int): Number of epochs between each evaluation. Default is 5.
        metrics (Sequence[Tuple[str, Dict[str, Any]]]): Metric names and their parameters to
            compute during evaluation. Default is ["hr", "ndcg"].
        model_loss_weight (float): Weight for the model-specific loss when combined with the
            recommendation loss. Default is 1.0.
        top_k (Sequence[int]): Cutoff values for computing top-K metrics during evaluation.
            Default is [1, 5, 10].
    """

    eval_interval: int = field(
        default=5,
        metadata={"help": "Number of epochs between each evaluation."},
    )

    metrics: Sequence[Tuple[str, Dict[str, Any]]] = field(
        default_factory=lambda: [
            ("hr", {}),
            ("ndcg", {}),
            ("popularity", {"p": (0.1, 0.2)}),
            ("unpopularity", {"p": (0.2, 0.4)}),
        ],
        metadata={"help": "List of metric names and their parameters to compute during evaluation."},
    )

    model_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for the model-specific loss when combined with the recommendation loss."},
    )

    top_k: Sequence[int] = field(
        default_factory=lambda: [1, 5, 10],
        metadata={"help": "Cutoff values for computing top-K metrics during evaluation."},
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
    batch_size: int = 4096,
) -> Dict[str, float]:
    """Compute metrics for sequential recommendation tasks.

    Args:
        prediction (EvalPrediction): Object containing model predictions and labels.
        train_dataset (SeqRecDataset): Dataset used during training; required for global metrics
            such as popularity-based measurements.
        top_k (Sequence[int]): Cutoff values for computing top-K metrics, determining how many
            predictions to consider for each metric.
        metrics (Sequence[Tuple[str, Dict[str, Any]]]): Metric specifications, where each tuple
            comprises the metric name and an optional parameter dictionary.
        device (Union[torch.device, str, None]): Device used for metric computations.
        batch_size (int): Number of samples per chunk when transferring logits/labels to the
            target device for top-K extraction.

    Returns:
        Dict[str, float]: Dictionary containing computed metric values keyed by metric name.

    .. note::
        As we may call this evaluation function for global metrics (e.g., popularity/fairness),
        you should ensure the `train_dataset` is provided if any global metrics are specified.
        In addition, `batch_eval_metrics` in `SeqRecTrainingArguments` should be set to `False`
        to avoid conflicts.
    """
    logits: Float[np.ndarray, "B I+1"]
    if isinstance(prediction.predictions, tuple):  # pragma: no cover - rarely used
        logits = prediction.predictions[0].astype(np.float32)
    else:
        logits = prediction.predictions.astype(np.float32)

    labels: Int[np.ndarray, "B L"]
    if isinstance(prediction.label_ids, tuple):  # pragma: no cover - rarely used
        labels = prediction.label_ids[0].astype(np.int64)
    else:
        labels = prediction.label_ids.astype(np.int64)
    last_step_labels: Int[np.ndarray, "B"] = labels[:, -1]

    num_items = logits.shape[1]
    max_k = min(max(top_k), num_items)
    torch_device = torch.device(device) if device is not None else torch.device("cpu")

    topk_chunks: List[Int[torch.Tensor, "b max_k"]] = []
    label_chunks: List[Int[torch.Tensor, "b"]] = []
    for start in range(0, logits.shape[0], batch_size):
        end = min(start + batch_size, logits.shape[0])
        logits_chunk = torch.from_numpy(logits[start:end]).to(torch_device)
        labels_chunk = torch.from_numpy(last_step_labels[start:end]).to(torch_device)
        _, indices = torch.topk(logits_chunk, k=max_k, dim=1)
        topk_chunks.append(indices)
        label_chunks.append(labels_chunk)

    topk_tensor = torch.cat(topk_chunks, dim=0)
    labels_tensor = torch.cat(label_chunks, dim=0)

    results: Dict[str, float] = {}
    for k in top_k:
        sliced_topk_indices = topk_tensor[:, :k]
        for metric_name, metric_params in metrics:
            metric_fn = MetricFactory.create(metric_name)
            metric_results = metric_fn(
                topk_indices=sliced_topk_indices,
                labels=labels_tensor,
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
            callbacks (Optional[List[TrainerCallback]]): Trainer callbacks (defaults to
                `EpochIntervalEvalCallback`).
            **kwargs (Any): Additional keyword arguments forwarded to the base `Trainer`.
        """
        try:
            first_param = next(model.parameters())
            model_device = first_param.device
        except StopIteration:  # pragma: no cover - models without parameters
            model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        eval_batch_size = getattr(args, "per_device_eval_batch_size", None)
        metric_batch_size = eval_batch_size * 4 if eval_batch_size else 4096

        if compute_metrics is None:
            compute_metrics = partial(
                compute_seqrec_metrics,
                train_dataset=train_dataset,
                top_k=args.top_k,
                metrics=args.metrics,
                device=model_device,
                batch_size=metric_batch_size,
            )

        if callbacks is None:
            callbacks = [EpochIntervalEvalCallback(eval_interval=args.eval_interval)]

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
                loss/logits when `return_outputs` is True. The loss combines the model-specific
                loss (if provided) with the recommendation loss computed via `compute_rec_loss`.
        """
        model = model.module if hasattr(model, "module") else model  # type: ignore - for distributed training

        outputs: SeqRecOutput = model(
            **inputs,
            output_model_loss=model.training,  # only compute model loss during training
        )
        assert isinstance(outputs, SeqRecOutput), "Model output must be an instance of SeqRecOutput."

        rec_loss = self.compute_rec_loss(inputs, outputs, num_items_in_batch)
        if outputs.model_loss is not None:
            loss = rec_loss + outputs.model_loss * self.args.model_loss_weight
        else:
            loss = rec_loss

        if return_outputs:
            last_step_hidden_states: Float[torch.Tensor, "B d"]
            last_step_hidden_states = outputs.last_hidden_state[:, -1, :]
            logits: Float[torch.Tensor, "B I+1"]
            logits = last_step_hidden_states @ model.item_embed.weight.T
            output_dict: Dict[str, torch.Tensor] = {
                "loss": loss,
                "logits": logits,
            }
            return loss, output_dict
        else:
            return loss

    @abstractmethod
    def compute_rec_loss(  # pragma: no cover - abstract method
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
                valid items in each sequence in the batch (excluding padding).

        Returns:
            Float[torch.Tensor, ""]: Computed recommendation loss as a scalar tensor.
        """
        ...
