"""Base trainer for sequential recommendation tasks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, Generic, List, Optional, Sequence, Tuple, Type, TypeVar, Union

from jaxtyping import Float, Int
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction
from transformers.training_args import TrainingArguments

from ...datasets import SeqRecCollator, SeqRecDataset
from ...models import SeqRecModel, SeqRecOutput
from .utils.callbacks import EpochIntervalEvalCallback, HardStopCallback
from .utils.evaluations import clip_top_k, compute_seqrec_metrics

__all__ = [
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


@dataclass
class SeqRecTrainingArguments(TrainingArguments):
    """Training arguments for sequential recommendation trainers.

    Args:
        norm_embeddings (bool): Whether to L2-normalize user and item embeddings during loss
            computation and evaluation. If True, both user and item embeddings are normalized to
            unit length, and the dot product is equivalent to cosine similarity. Default is False.
        eval_interval (int): Number of epochs between each evaluation. Default is 5.
        train_stop_epoch (int): Number of epochs to stop training. Default is -1 (no early stop).
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
    ) -> Union[Float[torch.Tensor, ""], Tuple[Float[torch.Tensor, ""], SeqRecOutput]]:
        """Computes the loss for a batch of inputs.

        Args:
            model (nn.Module): Model being trained.
            inputs (dict[str, Union[torch.Tensor, Any]]): Dictionary of input tensors, i.e., the
                `SeqRecCollator` output.
            return_outputs (bool): Whether to return the model outputs along with the loss.
            num_items_in_batch (Optional[torch.Tensor]): Optional tensor indicating the number of
                valid items in each sequence in the batch (excluding padding).

        Returns:
            Union[Float[torch.Tensor, ""], Tuple[Float[torch.Tensor, ""], SeqRecOutput]]:
                Either the scalar loss or a tuple containing the loss and the raw `SeqRecOutput`
                when `return_outputs` is True. The loss combines the model-specific loss (if provided)
                with the recommendation loss computed via `compute_rec_loss`.
        """
        model = model.module if hasattr(model, "module") else model  # type: ignore - for distributed training
        assert isinstance(model, SeqRecModel), "Model must be an instance of SeqRecModel."

        outputs: SeqRecOutput = model(
            **inputs,
            output_model_loss=model.training,  # only compute model loss during training
        )
        assert isinstance(outputs, SeqRecOutput), "Model output must be an instance of SeqRecOutput."

        rec_loss = self.compute_rec_loss(inputs, outputs, num_items_in_batch, self.args.norm_embeddings)
        if outputs.model_loss is not None:
            loss = rec_loss + outputs.model_loss * self.args.model_loss_weight
        else:
            loss = rec_loss

        if return_outputs:
            return loss, outputs
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
        This should be implemented by all subclasses.

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

    def _compute_topk_indices(
        self,
        outputs: SeqRecOutput,
        model: SeqRecModel[Any, Any],
    ) -> Int[torch.Tensor, "B K"]:
        """Compute top-K item predictions from the current model outputs."""

        last_step_hidden_states: Float[torch.Tensor, "B d"]
        last_step_hidden_states = outputs.last_hidden_state[:, -1, :]
        item_embed_weight: Float[torch.Tensor, "I+1 d"] = model.item_embed_weight

        if self.args.norm_embeddings:
            last_step_hidden_states = F.normalize(last_step_hidden_states, p=2, dim=-1)
            item_embed_weight = F.normalize(item_embed_weight, p=2, dim=-1)

        logits: Float[torch.Tensor, "B I+1"]
        logits = last_step_hidden_states @ item_embed_weight.T

        effective_top_k = max(1, min(self.max_top_k, self.item_size))
        _, topk_indices = torch.topk(logits, k=effective_top_k, dim=-1)  # may predict padding index

        return topk_indices

    def prediction_step(  # type: ignore[override]
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Evaluation forward pass that returns mean loss, top-K indices, and detached labels.

        Args:
            model (nn.Module): Model under evaluation. May be wrapped for distributed training.
            inputs (dict[str, Union[torch.Tensor, Any]]): Batch produced by :class:`SeqRecCollator`.
            prediction_loss_only (bool): Whether to skip prediction tensors and only surface the loss.
            ignore_keys (Optional[list[str]]): Unused placeholder required by :class:`~transformers.Trainer`.

        Returns:
            tuple[Optional[torch.Tensor], Optional[Int[torch.Tensor, "B K"]], Optional[Int[torch.Tensor, "B L"]]]:
                `(loss, topk_indices, labels)` where `loss` is the batch-averaged value, `topk_indices` matches
                `max_top_k` from the trainer config, and `labels` mirrors the collator's ground-truth tensor.
        """

        inputs = self._prepare_inputs(inputs)

        label_tensor = inputs[self.label_names[0]]
        assert isinstance(label_tensor, torch.Tensor), "Labels must be a tensor."
        labels: Int[torch.Tensor, "B L"] = label_tensor.detach()

        with torch.no_grad():
            num_items_in_batch = self._get_num_items_in_batch([inputs], self.args.device)
            loss, outputs = self.compute_loss(
                model,
                inputs,
                return_outputs=True,
                num_items_in_batch=num_items_in_batch,  # type: ignore - num_items_in_batch can be int
            )
            loss = loss.detach().mean() if loss is not None else None

        if prediction_loss_only:  # pragma: no cover - prediction loss only
            return loss, None, None

        unwrapped_model = model.module if hasattr(model, "module") else model  # type: ignore - distributed
        assert isinstance(unwrapped_model, SeqRecModel)
        assert isinstance(outputs, SeqRecOutput)
        predictions: Int[torch.Tensor, "B K"]
        predictions = self._compute_topk_indices(outputs, unwrapped_model).detach()

        return loss, predictions, labels
