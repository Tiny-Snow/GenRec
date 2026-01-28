"""Base trainer for quantizer models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, Generic, List, Optional, Sequence, Tuple, Type, TypeVar, Union

from jaxtyping import Float
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
    ) -> Union[Float[torch.Tensor, ""], Tuple[Float[torch.Tensor, ""], Dict[str, torch.Tensor]]]:
        """Computes the loss for a batch of inputs.

        Args:
            model (nn.Module): Model being trained.
            inputs (dict[str, Union[torch.Tensor, Any]]): Dictionary of input tensors, i.e., the
                `QuantizerCollator` output.
            return_outputs (bool): Whether to return the model outputs along with the loss.
            num_items_in_batch (Optional[torch.Tensor]): Optional tensor indicating the number of
                valid items in each sequence in the batch (excluding padding).

        Returns:
            Union[Float[torch.Tensor, ""], Tuple[Float[torch.Tensor, ""], Dict[str, torch.Tensor]]]:
                Either the scalar loss or a tuple containing the loss and a dictionary with
                loss and top-k indices of predicted items when `return_outputs` is True. The loss
                combines the model-specific loss (if provided) with the quantizer losses via
                `compute_quantizer_loss`.
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
            assert outputs.semantic_ids is not None, "Semantic IDs must be available in outputs."
            assert outputs.reconstruction_loss is not None, "Reconstruction loss should be available in outputs."
            assert outputs.codebook_loss is not None, "Codebook loss should be available in outputs."
            assert outputs.commitment_loss is not None, "Commitment loss should be available in outputs."
            assert "item_id" in inputs and isinstance(
                inputs["item_id"], torch.Tensor
            ), "Input batch must contain 'item_id' tensor."
            output_dict: Dict[str, torch.Tensor] = {
                "loss": loss,
                "semantic_ids": outputs.semantic_ids,
                "reconstruction_loss": outputs.reconstruction_loss,
                "codebook_loss": outputs.codebook_loss,
                "commitment_loss": outputs.commitment_loss,
                "item_id": inputs["item_id"],
            }
            return loss, output_dict
        else:
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
