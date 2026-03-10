"""Base trainer for generative recommendation models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, Generic, List, Optional, Sequence, Tuple, Type, TypeVar, Union

from jaxtyping import Float, Int
import torch
import torch.nn as nn
from transformers import Trainer
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.utils import GenerateBeamOutput
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction
from transformers.training_args import TrainingArguments

from ...datasets import GenRecCollator, GenRecDataset
from ...models import GenRecModel, GenRecOutput
from .utils.callbacks import EpochIntervalEvalCallback, HardStopCallback
from .utils.evaluations import clip_top_k, compute_genrec_metrics


__all__ = [
    "GenRecTrainer",
    "GenRecTrainerFactory",
    "GenRecTrainingArguments",
    "GenRecTrainingArgumentsFactory",
]

_GenRecModel = TypeVar("_GenRecModel", bound="GenRecModel[Any, Any, Any]")
_GenRecTrainer = TypeVar("_GenRecTrainer", bound="GenRecTrainer[Any, Any]")
_GenRecTrainingArguments = TypeVar("_GenRecTrainingArguments", bound="GenRecTrainingArguments")


class GenRecTrainingArgumentsFactory:  # pragma: no cover - factory class
    """Factory for creating `GenRecTrainingArguments` instances."""

    _registry: dict[str, Type[GenRecTrainingArguments]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Type[_GenRecTrainingArguments]], Type[_GenRecTrainingArguments]]:
        """Decorator to register a `GenRecTrainingArguments` implementation."""

        def decorator(
            training_args_cls: Type[_GenRecTrainingArguments],
        ) -> Type[_GenRecTrainingArguments]:
            if name in cls._registry:
                raise ValueError(f"GenRec training arguments '{name}' is already registered.")
            cls._registry[name] = training_args_cls
            return training_args_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs) -> GenRecTrainingArguments:
        """Creates an instance of a registered `GenRecTrainingArguments`."""
        if name not in cls._registry:
            raise ValueError(f"GenRec training arguments '{name}' is not registered.")
        training_args_cls = cls._registry[name]
        return training_args_cls(**kwargs)


@dataclass
class GenRecTrainingArguments(TrainingArguments):
    """Training arguments for generative recommendation trainers.

    Args:
        eval_interval (int): Number of epochs between each evaluation. Default is 5.
        train_stop_epoch (int): Number of epochs to stop training. Default is -1 (no early stop).
        metrics (Sequence[Tuple[str, Dict[str, Any]]]): Metric names and their parameters to
            compute during evaluation. Default is [('hr', {}), ('ndcg', {}), ('popularity', {'p': (0.1, 0.2)}),
            ("unpopularity", {"p": (0.2, 0.4)})].
        model_loss_weight (float): Weight for the model-specific loss when combined with the
            recommendation loss. Default is 0.0.
        top_k (Sequence[int]): Cutoff values for computing top-K metrics during evaluation.
            Default is [1, 5, 10].
        num_beams (int): Number of beams for beam search during evaluation. It should be greater than or equal to
            the maximum value in `top_k`. Default is 10.
    """

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

    num_beams: int = field(
        default=10,
        metadata={
            "help": (
                "Number of beams for beam search during evaluation. It should be greater than or equal to "
                "the maximum value in `top_k`. Default is 10."
            )
        },
    )


class GenRecTrainerFactory:  # pragma: no cover - factory class
    """Factory for creating `GenRecTrainer` instances."""

    _registry: dict[str, Type[GenRecTrainer[Any, Any]]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Type[_GenRecTrainer]], Type[_GenRecTrainer]]:
        """Decorator to register a `GenRecTrainer` implementation."""

        def decorator(trainer_cls: Type[_GenRecTrainer]) -> Type[_GenRecTrainer]:
            if name in cls._registry:
                raise ValueError(f"GenRec trainer '{name}' is already registered.")
            cls._registry[name] = trainer_cls
            return trainer_cls

        return decorator

    @classmethod
    def create(
        cls,
        name: str,
        model: GenRecModel[Any, Any, Any],
        args: GenRecTrainingArguments,
        data_collator: GenRecCollator,
        train_dataset: GenRecDataset,
        eval_dataset: Optional[GenRecDataset] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        **kwargs,
    ) -> GenRecTrainer[Any, Any]:
        """Creates an instance of a registered `GenRecTrainer`."""
        if name not in cls._registry:
            raise ValueError(f"GenRec trainer '{name}' is not registered.")
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


class GenRecTrainer(Trainer, Generic[_GenRecModel, _GenRecTrainingArguments], ABC):
    """Base trainer class for generative recommendation models.

    This class extends the `Trainer` class from the `transformers` library. You should
    implement specific training logic, i.e., `compute_loss`, in subclasses to
    compute the model-agnostic loss for generative recommendation tasks.

    .. note::
        We set up the default callbacks to include `EpochIntervalEvalCallback`
        which performs evaluation every `eval_interval` epochs (default is 5).

    .. note::
        We also set up the `compute_metrics` function to use `compute_genrec_metrics` by default.
    """

    args: _GenRecTrainingArguments
    model: _GenRecModel

    def __init__(
        self,
        model: _GenRecModel,
        args: _GenRecTrainingArguments,
        data_collator: GenRecCollator,
        train_dataset: GenRecDataset,
        eval_dataset: Optional[GenRecDataset] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        **kwargs,
    ) -> None:
        """Initializes the GenRecTrainer with the given model and training arguments.

        Args:
            model (_GenRecModel): Generative recommendation model to be trained.
            args (_GenRecTrainingArguments): Training arguments specific to generative recommendation.
            data_collator (GenRecCollator): Data collator that prepares model inputs.
            train_dataset (GenRecDataset): Dataset used for training.
            eval_dataset (Optional[GenRecDataset]): Dataset used for evaluation.
            compute_metrics (Optional[Callable[[EvalPrediction], Dict]]): Function used to compute
                metrics during evaluation. Defaults to :func:`compute_genrec_metrics`.
            callbacks (Optional[List[TrainerCallback]]): Trainer callbacks. Defaults to
                `[EpochIntervalEvalCallback, HardStopCallback]`.
            **kwargs (Any): Additional keyword arguments forwarded to the base `Trainer`.
        """
        self.item_size = train_dataset.item_size
        self.top_k = tuple(clip_top_k(args.top_k, self.item_size))
        self.max_top_k = max(self.top_k)

        # sid_width is used as the number of new_tokens for generation
        assert train_dataset.sid_width is not None, "sid_width must be defined in the train_dataset."
        self.sid_width = train_dataset.sid_width

        # num_beams must be >= max_top_k for generation
        self.num_beams = args.num_beams
        assert args.num_beams >= self.max_top_k, (
            f"num_beams ({args.num_beams}) must be greater than or equal to "
            f"the maximum value in top_k ({self.max_top_k})."
        )

        # Set <PAD>, <EOS>, and decoder_start_token_id from model config
        self.pad_token_id = model.config.pad_token_id
        self.eos_token_id = model.config.eos_token_id
        self.decoder_start_token_id = model.config.decoder_start_token_id
        self.use_cache = model.config.use_cache

        # Prepare GenerationConfig and prefix_allowed_tokens_fn for generation during evaluation
        self.gen_cfg = GenerationConfig(
            max_new_tokens=self.sid_width,
            min_new_tokens=self.sid_width,
            early_stopping=True,
            num_beams=self.num_beams,
            use_cache=self.use_cache,
            num_return_sequences=self.num_beams,
            return_dict_in_generate=True,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
            decoder_start_token_id=self.decoder_start_token_id,
        )
        self.prefix_allowed_tokens_fn = train_dataset.get_prefix_allowed_tokens_fn()

        try:
            first_param = next(model.parameters())
            model_device = first_param.device
        except StopIteration:  # pragma: no cover - models without parameters
            model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if compute_metrics is None:
            compute_metrics = partial(
                compute_genrec_metrics,
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
    ) -> Union[Float[torch.Tensor, ""], Tuple[Float[torch.Tensor, ""], GenRecOutput]]:
        """Computes the loss for a batch of inputs.

        Args:
            model (nn.Module): Model being trained.
            inputs (dict[str, Union[torch.Tensor, Any]]): Dictionary of input tensors, i.e., the
                `GenRecCollator` output.
            return_outputs (bool): Whether to return the model outputs along with the loss.
            num_items_in_batch (Optional[torch.Tensor]): Optional tensor indicating the number of
                valid items in each sequence in the batch (excluding padding).

        Returns:
            Union[Float[torch.Tensor, ""], Tuple[Float[torch.Tensor, ""], GenRecOutput]]:
                Either the scalar loss or a tuple containing the loss and the raw `GenRecOutput`
                when `return_outputs` is True. The loss combines the model-specific loss (if provided)
                with the recommendation loss computed via `compute_rec_loss`.
        """
        model = model.module if hasattr(model, "module") else model  # type: ignore - for distributed training
        assert isinstance(model, GenRecModel), "Model must be an instance of GenRecModel."

        outputs: GenRecOutput = model(
            **inputs,
            output_model_loss=model.training,  # only compute model loss during training
        )
        assert isinstance(outputs, GenRecOutput), "Model output must be an instance of GenRecOutput."

        rec_loss = self.compute_rec_loss(inputs, outputs, num_items_in_batch)
        if outputs.model_loss is not None:
            loss = rec_loss + outputs.model_loss * self.args.model_loss_weight
        else:  # pragma: no cover - no model loss
            loss = rec_loss

        if return_outputs:
            return loss, outputs
        return loss

    @abstractmethod
    def compute_rec_loss(  # pragma: no cover - abstract method
        self,
        inputs: dict[str, Union[torch.Tensor, Any]],
        outputs: GenRecOutput,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ) -> Float[torch.Tensor, ""]:
        """Computes the recommendation loss for a batch of inputs and model outputs.
        This should be implemented by all subclasses.

        Args:
            inputs (dict[str, Union[torch.Tensor, Any]]): Dictionary of input tensors, i.e., the
                `GenRecCollator` output.
            outputs (GenRecOutput): Model outputs from the forward pass.
            num_items_in_batch (Optional[torch.Tensor]): Optional tensor indicating the number of
                valid items in each sequence in the batch (excluding padding).

        Returns:
            Float[torch.Tensor, ""]: Computed recommendation loss as a scalar tensor.
        """
        ...

    def prediction_step(  # type: ignore[override]
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
    ) -> Tuple[
        Optional[Float[torch.Tensor, ""]],
        Optional[Int[torch.Tensor, "B num_beams C"]],
        Optional[Int[torch.Tensor, "B C"]],
    ]:
        """Evaluation forward pass that returns mean loss, top-K predicted SIDs, and ground-truth item sids.

        Args:
            model (nn.Module): Model under evaluation. May be wrapped for distributed training.
            inputs (dict[str, Union[torch.Tensor, Any]]): Batch produced by :class:`GenRecCollator`.
            prediction_loss_only (bool): Whether to skip prediction tensors and only surface the loss.
            ignore_keys (Optional[list[str]]): Unused placeholder required by :class:`~transformers.Trainer`.

        Returns:
            Tuple[
                Optional[Float[torch.Tensor, ""]],
                Optional[Int[torch.Tensor, "B num_beams C"]],
                Optional[Int[torch.Tensor, "B C"]],
            ]: A tuple containing:
            - Mean loss over the batch as a scalar tensor.
            - Predicted SIDs tensor of shape (batch_size, num_beams, sid_width)
            - Ground-truth SIDs tensor of shape (batch_size, sid_width).
        """

        inputs = self._prepare_inputs(inputs)

        label_tensor = inputs[self.label_names[0]]
        assert isinstance(label_tensor, torch.Tensor), "Labels must be a tensor."
        labels: Int[torch.Tensor, "B C+1"] = label_tensor.detach()

        # Compute loss
        with torch.no_grad():
            num_items_in_batch = self._get_num_items_in_batch([inputs], self.args.device)
            loss, forward_outputs = self.compute_loss(
                model,
                inputs,
                return_outputs=True,
                num_items_in_batch=num_items_in_batch,  # type: ignore - num_items_in_batch can be int
            )
            loss = loss.detach().mean() if loss is not None else None
            assert isinstance(forward_outputs, GenRecOutput)

        if prediction_loss_only:  # pragma: no cover - prediction loss only
            return loss, None, None

        # Generate predictions
        unwrapped_model = model.module if hasattr(model, "module") else model  # type: ignore - distributed
        assert isinstance(unwrapped_model, GenRecModel)

        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)

        generation_outputs = unwrapped_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=self.gen_cfg,
            prefix_allowed_tokens_fn=self.prefix_allowed_tokens_fn,
        )
        assert isinstance(generation_outputs, GenerateBeamOutput)

        # Get predicted sids, scores, and calculate metrics
        batch_size = input_ids.size(0)
        predictions: Int[torch.Tensor, "B num_beams C"]
        predictions = generation_outputs.sequences[:, 1:].reshape(batch_size, self.num_beams, self.sid_width)
        labels = labels[:, :-1]  # remove eos token for labels

        return loss, predictions, labels
