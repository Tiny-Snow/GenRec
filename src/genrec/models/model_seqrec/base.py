"""Base model for sequential recommendation tasks."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Generic, Optional, Tuple, Type, TypeVar, Union

from jaxtyping import Float, Int
import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.utils.generic import ModelOutput

__all__ = [
    "SeqRecModel",
    "SeqRecModelFactory",
    "SeqRecModelConfig",
    "SeqRecModelConfigFactory",
    "SeqRecOutput",
    "SeqRecOutputFactory",
]


_SeqRecModelConfig = TypeVar("_SeqRecModelConfig", bound="SeqRecModelConfig")
_SeqRecOutput = TypeVar("_SeqRecOutput", bound="SeqRecOutput")
_SeqRecModel = TypeVar("_SeqRecModel", bound="SeqRecModel[Any, Any]")


class SeqRecModelConfigFactory:  # pragma: no cover - factory class
    """Factory for creating `SeqRecModelConfig` instances."""

    _registry: dict[str, Type[SeqRecModelConfig]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Type[_SeqRecModelConfig]], Type[_SeqRecModelConfig]]:
        """Decorator to register a `SeqRecModelConfig` implementation."""

        def decorator(config_cls: Type[_SeqRecModelConfig]) -> Type[_SeqRecModelConfig]:
            if name in cls._registry:
                raise ValueError(f"SeqRec model config '{name}' is already registered.")
            cls._registry[name] = config_cls
            return config_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs) -> SeqRecModelConfig:
        """Creates an instance of a registered `SeqRecModelConfig`."""
        if name not in cls._registry:
            raise ValueError(f"SeqRec model config '{name}' is not registered.")
        config_cls = cls._registry[name]
        return config_cls(**kwargs)


class SeqRecModelFactory:  # pragma: no cover - factory class
    """Factory for creating `SeqRecModel` instances."""

    _registry: dict[str, Type[SeqRecModel[Any, Any]]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Type[_SeqRecModel]], Type[_SeqRecModel]]:
        """Decorator to register a `SeqRecModel` implementation."""

        def decorator(model_cls: Type[_SeqRecModel]) -> Type[_SeqRecModel]:
            if name in cls._registry:
                raise ValueError(f"SeqRec model '{name}' is already registered.")
            cls._registry[name] = model_cls
            return model_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs) -> SeqRecModel[Any, Any]:
        """Creates an instance of a registered `SeqRecModel`."""
        if name not in cls._registry:
            raise ValueError(f"SeqRec model '{name}' is not registered.")
        model_cls = cls._registry[name]
        return model_cls(**kwargs)

    @classmethod
    def from_pretrained(cls, name: str, path: Union[str, os.PathLike], **kwargs) -> SeqRecModel[Any, Any]:
        """Loads a pretrained instance of a registered `SeqRecModel`."""
        if name not in cls._registry:
            raise ValueError(f"SeqRec model '{name}' is not registered.")
        model_cls = cls._registry[name]
        return model_cls.from_pretrained(path, **kwargs)


class SeqRecOutputFactory:  # pragma: no cover - factory class
    """Factory for creating `SeqRecOutput` instances."""

    _registry: dict[str, Type[SeqRecOutput]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Type[_SeqRecOutput]], Type[_SeqRecOutput]]:
        """Decorator to register a `SeqRecOutput` implementation."""

        def decorator(output_cls: Type[_SeqRecOutput]) -> Type[_SeqRecOutput]:
            if name in cls._registry:
                raise ValueError(f"SeqRec output '{name}' is already registered.")
            cls._registry[name] = output_cls
            return output_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs) -> SeqRecOutput:
        """Creates an instance of a registered `SeqRecOutput`."""
        if name not in cls._registry:
            raise ValueError(f"SeqRec output '{name}' is not registered.")
        output_cls = cls._registry[name]
        return output_cls(**kwargs)


class SeqRecModelConfig(PretrainedConfig):
    """Base configuration class for sequential recommendation models.

    This class extends the `PretrainedConfig` from the Hugging Face Transformers library
    and serves as a base for implementing specific sequential recommendation model configurations.

    Subclasses must specify the `model_type` attribute.
    """

    model_type = "seqrec"

    def __init__(
        self,
        item_size: int = 1024,
        hidden_size: int = 256,
        num_attention_heads: int = 4,
        num_hidden_layers: int = 4,
        **kwargs,
    ) -> None:
        """Initializes the configuration with model hyperparameters.

        Args:
            item_size (int): Size of the item vocabulary, excluding the padding token (0-th).
            hidden_size (int): Dimensionality of the model's hidden representations.
            num_attention_heads (int): Number of attention heads in the model.
            num_hidden_layers (int): Number of hidden layers in the model.
            **kwargs (Any): Additional keyword arguments for the base `PretrainedConfig`.
        """
        super().__init__(**kwargs)

        self.item_size = item_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers


@dataclass
class SeqRecOutput(ModelOutput):
    """Base output class for sequential recommendation models.

    Attributes:
        last_hidden_state: Hidden states from the last layer of the model.
            The shape is (batch_size, seq_len, hidden_size).
        model_loss: The computed model-specific loss value, if applicable.
            Note that the model-agnostic loss (e.g., NLL loss) is handled outside
            of this class.
        hidden_states: Hidden states from the model, if applicable. The shape is
            (batch_size, seq_len, hidden_size).
        attentions: Attention weights from the model, if applicable. The shape is
            (batch_size, num_heads, seq_len, seq_len).
    """

    last_hidden_state: Float[torch.Tensor, "B L d"]
    model_loss: Optional[Float[torch.Tensor, ""]] = None
    hidden_states: Optional[Tuple[Float[torch.Tensor, "B L d"], ...]] = None
    attentions: Optional[Tuple[Float[torch.Tensor, "B H L L"], ...]] = None


class SeqRecModel(PreTrainedModel, Generic[_SeqRecModelConfig, _SeqRecOutput], ABC):
    """Base class for sequential recommendation models.

    This class extends the `PreTrainedModel` from the Hugging Face Transformers library
    and serves as a base for implementing specific sequential recommendation models.

    Subclasses must specify the `config_class` attribute and implement the `forward` method.
    """

    config_class: Type[_SeqRecModelConfig]
    supports_gradient_checkpointing = True

    def __init__(self, config: _SeqRecModelConfig) -> None:
        """Initializes the sequential recommendation model.

        Args:
            config (_SeqRecModelConfig): Configuration containing model hyperparameters.
        """
        super().__init__(config)
        self.config: _SeqRecModelConfig
        self._item_embed = nn.Embedding(config.item_size + 1, config.hidden_size, padding_idx=0)

    def embed_tokens(
        self,
        input_ids: Int[torch.Tensor, "B L"],
    ) -> Float[torch.Tensor, "B L d"]:
        """Embeds input item ID sequences.

        Args:
            input_ids (Int[torch.Tensor, "B L"]): Input item ID sequences of shape (batch_size, seq_len).

        Returns:
            Float[torch.Tensor, "B L d"]: Embedded item representations of shape (batch_size, seq_len, hidden_size).
        """
        return self._item_embed(input_ids)

    @property
    def item_embed_weight(self) -> Float[torch.Tensor, "I+1 d"]:
        """Returns the item embedding weight matrix.

        Returns:
            Float[torch.Tensor, "I+1 d"]: Item embedding weight matrix of shape (item_size + 1, hidden_size).
        """
        return self._item_embed.weight

    @property
    def item_size(self) -> int:
        """Returns the size of the item vocabulary (excluding padding token)."""
        return self.config.item_size

    def _set_gradient_checkpointing(
        self,
        enable: bool = True,
        gradient_checkpointing_func: Optional[Callable[..., torch.Tensor]] = None,
    ) -> None:
        """Hooks into HF's gradient checkpointing toggles by tracking the enable flag."""

        self.gradient_checkpointing = enable
        if gradient_checkpointing_func is not None:
            self._gradient_checkpointing_func = gradient_checkpointing_func  # pragma: no cover - optional hook

    @abstractmethod
    def forward(  # pragma: no cover - abstract method
        self,
        input_ids: Int[torch.Tensor, "B L"],
        attention_mask: Int[torch.Tensor, "B L"],
        output_model_loss: bool = False,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        **kwargs,
    ) -> _SeqRecOutput:
        """Defines the forward pass of the sequential recommendation model.

        Args:
            input_ids (Int[torch.Tensor, "B L"]): Input item ID sequences of shape (batch_size, seq_len).
            attention_mask (Int[torch.Tensor, "B L"]): Attention masks for inputs.
            output_model_loss (bool): Whether to compute and return the model-specific loss. Default is False.
            output_hidden_states (bool): Whether to return hidden states from all layers. Default is False.
            output_attentions (bool): Whether to return attention weights from all layers. Default is False.
            **kwargs (Any): Additional keyword arguments for the model.

        Returns:
            _SeqRecOutput: Model outputs packaged as a `SeqRecOutput` instance.
        """
        ...
