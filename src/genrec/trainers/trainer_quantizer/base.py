"""Base trainer for quantizer models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, Generic, List, Optional, Sequence, Tuple, Type, TypeVar, Union

from jaxtyping import Float, Int
import torch
import torch.nn as nn
from transformers import EvalPrediction, Trainer, TrainerCallback, TrainingArguments

from ...datasets import QuantizerCollator, QuantizerDataset
from ...models import QuantizerModel, QuantizerOutput
from .utils.callbacks import EpochIntervalEvalCallback, HardStopCallback
from .utils.evaluations import compute_quantizer_metrics

__all__ = [
    "QuantizerTrainer",
    "QuantizerTrainerFactory",
    "QuantizerTrainingArguments",
    "QuantizerTrainingArgumentsFactory",
]


_QuantizerModel = TypeVar("_QuantizerModel", bound="QuantizerModel[Any, Any]")
_QuantizerTrainer = TypeVar("_QuantizerTrainer", bound="QuantizerTrainer[Any, Any]")
_QuantizerTrainingArguments = TypeVar("_QuantizerTrainingArguments", bound="QuantizerTrainingArguments")


class QuantizerTrainingArgumentsFactory:  # pragma: no cover - factory class
    """Factory for creating `QuantizerTrainingArguments` instances."""

    _registry: dict[str, Type[QuantizerTrainingArguments]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Type[_QuantizerTrainingArguments]], Type[_QuantizerTrainingArguments]]:
        """Decorator to register a `QuantizerTrainingArguments` implementation."""

        def decorator(
            training_args_cls: Type[_QuantizerTrainingArguments],
        ) -> Type[_QuantizerTrainingArguments]:
            if name in cls._registry:
                raise ValueError(f"Quantizer training arguments '{name}' is already registered.")
            cls._registry[name] = training_args_cls
            return training_args_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs) -> QuantizerTrainingArguments:
        """Creates an instance of a registered `QuantizerTrainingArguments`."""
        if name not in cls._registry:
            raise ValueError(f"Quantizer training arguments '{name}' is not registered.")
        training_args_cls = cls._registry[name]
        return training_args_cls(**kwargs)


@dataclass
class QuantizerTrainingArguments(TrainingArguments):
    """Training arguments for quantizer trainers.

    Args:
        eval_interval (int): Number of epochs between each evaluation. Default is 100.
        train_stop_epoch (int): Number of epochs to stop training. Default is -1 (no early stop).
        metrics (Sequence[Tuple[str, Dict[str, Any]]]): Metric names and their parameters to
            compute during evaluation. Default is [('codebook_usage', {}), ('code_collision', {})].
        codebook_loss_weight (float): Weight for the codebook loss term. Default is 1.0.
        commitment_loss_weight (float): Weight for the commitment loss term. Default is 0.25.
        model_loss_weight (float): Weight for the model-specific loss. Default is 0.0.
    """

    eval_interval: int = field(
        default=100,
        metadata={"help": "Number of epochs between each evaluation. Default is 100."},
    )

    train_stop_epoch: int = field(
        default=-1,
        metadata={"help": "Number of epochs to stop training. Default is -1 (no early stop)."},
    )

    metrics: Sequence[Tuple[str, Dict[str, Any]]] = field(
        default_factory=lambda: [
            ("codebook_usage", {}),
            ("code_collision", {}),
        ],
        metadata={
            "help": (
                "Metric names and their parameters to compute during evaluation. "
                "Default is [('codebook_usage', {}), ('code_collision', {})]."
            )
        },
    )

    codebook_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for the codebook loss term. Default is 1.0."},
    )

    commitment_loss_weight: float = field(
        default=0.25,
        metadata={"help": "Weight for the commitment loss term. Default is 0.25."},
    )

    model_loss_weight: float = field(
        default=0.0,
        metadata={"help": "Weight for the model-specific loss. Default is 0.0."},
    )


class QuantizerTrainerFactory:  # pragma: no cover - factory class
    """Factory for creating `QuantizerTrainer` instances."""

    _registry: dict[str, Type[QuantizerTrainer[Any, Any]]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Type[_QuantizerTrainer]], Type[_QuantizerTrainer]]:
        """Decorator to register a `QuantizerTrainer` implementation."""

        def decorator(trainer_cls: Type[_QuantizerTrainer]) -> Type[_QuantizerTrainer]:
            if name in cls._registry:
                raise ValueError(f"Quantizer trainer '{name}' is already registered.")
            cls._registry[name] = trainer_cls
            return trainer_cls

        return decorator

    @classmethod
    def create(
        cls,
        name: str,
        model: QuantizerModel[Any, Any],
        args: QuantizerTrainingArguments,
        data_collator: QuantizerCollator,
        train_dataset: QuantizerDataset,
        eval_dataset: Optional[QuantizerDataset] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        **kwargs,
    ) -> QuantizerTrainer[Any, Any]:
        """Creates an instance of a registered `QuantizerTrainer`."""
        if name not in cls._registry:
            raise ValueError(f"Quantizer trainer '{name}' is not registered.")
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


class QuantizerTrainer(Trainer, Generic[_QuantizerModel, _QuantizerTrainingArguments], ABC):
    """Base trainer class for quantizer models.

    This class extends the `Trainer` class from the `transformers` library. You should
    implement specific training logic, i.e., `compute_loss`, in subclasses to
    compute the model-agnostic loss for quantizer training.

    .. note::
        We set up the default callbacks to include `EpochIntervalEvalCallback`
        which performs evaluation every `eval_interval` epochs (default is 100).

    .. note::
        We also set up the `compute_metrics` function to use `compute_quantizer_metrics` by default.
    """

    args: _QuantizerTrainingArguments
    model: _QuantizerModel

    def __init__(
        self,
        model: _QuantizerModel,
        args: _QuantizerTrainingArguments,
        data_collator: QuantizerCollator,
        train_dataset: QuantizerDataset,
        eval_dataset: Optional[QuantizerDataset] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        **kwargs,
    ) -> None:
        """Initializes the QuantizerTrainer with the given model and training arguments.

        Args:
            model (_QuantizerModel): Quantizer to be trained.
            args (_QuantizerTrainingArguments): Training arguments specific to quantizer training.
            data_collator (QuantizerCollator): Data collator that prepares model inputs.
            train_dataset (QuantizerDataset): Dataset used for training.
            eval_dataset (Optional[QuantizerDataset]): Dataset used for evaluation.
            compute_metrics (Optional[Callable[[EvalPrediction], Dict]]): Function used to compute
                metrics during evaluation. Defaults to :func:`compute_quantizer_metrics`.
            callbacks (Optional[List[TrainerCallback]]): Trainer callbacks. Defaults to
                `[EpochIntervalEvalCallback, HardStopCallback]`.
            **kwargs (Any): Additional keyword arguments forwarded to the base `Trainer`.
        """
        if compute_metrics is None:
            compute_metrics = partial(
                compute_quantizer_metrics,
                train_dataset=train_dataset,
                metrics=args.metrics,
                codebook_size=model.config.codebook_size,
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
        # we set it to ["item_id"] by default.
        # Your model's forward method and data collator should ensure that
        # the input batch contains a key "item_id" corresponding to the ground truth labels.
        # You may override this attribute if your label key is different in subclasses.
        self.label_names = ["item_id"]

        # initialize codebooks
        self.initialize_codebooks()

    def initialize_codebooks(self) -> None:
        """Initializes the quantizer codebooks before training.
        You may override this method in subclasses if custom initialization is needed.
        """
        model = self.model.module if hasattr(self.model, "module") else self.model  # type: ignore - for distributed training
        assert isinstance(model, QuantizerModel), "Model must be an instance of QuantizerModel."

        assert isinstance(
            self.train_dataset, QuantizerDataset
        ), "Train dataset must be an instance of QuantizerDataset."
        item_embeddings = self.train_dataset.item_textual_embeddings
        assert item_embeddings is not None, "Item embeddings are required to initialize codebooks."
        model.initialize_codebooks(torch.from_numpy(item_embeddings).to(model.device))

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ) -> Union[Float[torch.Tensor, ""], Tuple[Float[torch.Tensor, ""], QuantizerOutput]]:
        """Computes the loss for a batch of inputs.

        Args:
            model (nn.Module): Model being trained.
            inputs (dict[str, Union[torch.Tensor, Any]]): Dictionary of input tensors, i.e., the
                `QuantizerCollator` output.
            return_outputs (bool): Whether to return the model outputs along with the loss.
            num_items_in_batch (Optional[torch.Tensor]): Optional tensor indicating the number of
                valid items in each sequence in the batch (excluding padding).

        Returns:
            Union[Float[torch.Tensor, ""], Tuple[Float[torch.Tensor, ""], QuantizerOutput]]:
                Either the scalar loss or a tuple containing the loss and the raw `QuantizerOutput`
                when `return_outputs` is True. The loss combines the model-specific loss (if provided)
                with the quantizer losses via `compute_quantizer_loss`.
        """
        model = model.module if hasattr(model, "module") else model  # type: ignore - for distributed training
        assert isinstance(model, QuantizerModel), "Model must be an instance of QuantizerModel."

        outputs: QuantizerOutput = model(
            **inputs,
            output_loss=True,
            output_model_loss=model.training,  # only compute model loss during training
            output_embeddings=False,  # no need to output embeddings in Trainer
        )
        assert isinstance(outputs, QuantizerOutput), "Model output must be an instance of QuantizerOutput."

        quantizer_loss = self.compute_quantizer_loss(inputs, outputs, num_items_in_batch)
        if outputs.model_loss is not None:
            loss = quantizer_loss + outputs.model_loss * self.args.model_loss_weight
        else:
            loss = quantizer_loss

        if return_outputs:
            return loss, outputs
        return loss

    @abstractmethod
    def compute_quantizer_loss(  # pragma: no cover - abstract method
        self,
        inputs: dict[str, Union[torch.Tensor, Any]],
        outputs: QuantizerOutput,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ) -> Float[torch.Tensor, ""]:
        """Computes the model-agnostic quantizer loss for a batch of inputs and model outputs.

        This method should be implemented by all subclasses to compute the quantizer-specific loss
        components, e.g., reconstruction loss, codebook loss, and commitment loss.

        Args:
            inputs (dict[str, Union[torch.Tensor, Any]]): Dictionary of input tensors, i.e., the
                `QuantizerCollator` output.
            outputs (QuantizerOutput): Output from the quantizer model's forward pass.
            num_items_in_batch (Optional[torch.Tensor]): Optional tensor indicating the number of
                valid items in each sequence in the batch (excluding padding).

        Returns:
            Float[torch.Tensor, ""]: Scalar tensor representing the computed quantizer loss.
        """
        pass

    def _build_prediction_outputs(
        self,
        outputs: QuantizerOutput,
        inputs: dict[str, Union[torch.Tensor, Any]],
    ) -> tuple[torch.Tensor, ...]:
        """Package tensors required by quantizer metrics into a tuple."""

        assert outputs.semantic_ids is not None, "Semantic IDs must be available in outputs."
        assert outputs.reconstruction_loss is not None, "Reconstruction loss should be available in outputs."
        assert outputs.codebook_loss is not None, "Codebook loss should be available in outputs."
        assert outputs.commitment_loss is not None, "Commitment loss should be available in outputs."
        item_id = inputs.get("item_id")
        assert isinstance(item_id, torch.Tensor), "Input batch must contain 'item_id' tensor."

        semantic_ids: Int[torch.Tensor, "B C"] = outputs.semantic_ids.detach()
        reconstruction_loss: Float[torch.Tensor, "B"] = outputs.reconstruction_loss.detach()
        codebook_loss: Float[torch.Tensor, "B"] = outputs.codebook_loss.detach()
        commitment_loss: Float[torch.Tensor, "B"] = outputs.commitment_loss.detach()
        item_id_tensor: Int[torch.Tensor, "B"] = item_id.detach()

        return (semantic_ids, reconstruction_loss, codebook_loss, commitment_loss, item_id_tensor)

    def prediction_step(  # type: ignore[override]
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[tuple[torch.Tensor, ...]], Optional[torch.Tensor]]:
        """Evaluation step that surfaces loss plus the semantic-ID payloads required by the metrics.

        Args:
            model (nn.Module): Quantizer under evaluation. May be wrapped by `nn.DataParallel`/`nn.DistributedDataParallel`.
            inputs (dict[str, Union[torch.Tensor, Any]]): Batch prepared by :class:`QuantizerCollator`.
            prediction_loss_only (bool): Whether to suppress prediction tensors and only output the loss.
            ignore_keys (Optional[list[str]]): Present for :class:`~transformers.Trainer` compatibility; unused.

        Returns:
            tuple[Optional[torch.Tensor], Optional[tuple[torch.Tensor, ...]], Optional[Int[torch.Tensor, "B"]]]:
                `(loss, payload, labels)` where `payload` matches the tuple expected by
                :func:`compute_quantizer_metrics` and `labels` is just the detached `item_id` tensor.
        """

        inputs = self._prepare_inputs(inputs)

        label_tensor = inputs[self.label_names[0]]
        assert isinstance(label_tensor, torch.Tensor), "Item IDs must be a tensor."
        labels: Int[torch.Tensor, "B"] = label_tensor.detach()

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

        assert isinstance(outputs, QuantizerOutput)
        predictions = tuple(t.detach() for t in self._build_prediction_outputs(outputs, inputs))

        return loss, predictions, labels
