"""Base model for generative recommendation tasks."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, Optional, Type, TypeVar, Union

from jaxtyping import Float, Int
import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.cache_utils import Cache
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, Seq2SeqLMOutput

__all__ = [
    "GenRecModel",
    "GenRecModelFactory",
    "GenRecModelConfig",
    "GenRecModelConfigFactory",
    "GenRecOutput",
    "GenRecOutputFactory",
    "ShiftRightMixin",
]


_GenRecModelConfig = TypeVar("_GenRecModelConfig", bound="GenRecModelConfig")
_GenRecOutput = TypeVar("_GenRecOutput", bound="GenRecOutput")
_GenRecEncoderDecoderOutput = TypeVar("_GenRecEncoderDecoderOutput", bound=BaseModelOutputWithPastAndCrossAttentions)
_GenRecModel = TypeVar("_GenRecModel", bound="GenRecModel[Any, Any, Any]")


class GenRecModelConfigFactory:  # pragma: no cover - factory class
    """Factory for creating `GenRecModelConfig` instances."""

    _registry: dict[str, Type[GenRecModelConfig]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Type[_GenRecModelConfig]], Type[_GenRecModelConfig]]:
        """Decorator to register a `GenRecModelConfig` implementation."""

        def decorator(config_cls: Type[_GenRecModelConfig]) -> Type[_GenRecModelConfig]:
            if name in cls._registry:
                raise ValueError(f"GenRec model config '{name}' is already registered.")
            cls._registry[name] = config_cls
            return config_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs) -> GenRecModelConfig:
        """Creates an instance of a registered `GenRecModelConfig`."""
        if name not in cls._registry:
            raise ValueError(f"GenRec model config '{name}' is not registered.")
        config_cls = cls._registry[name]
        return config_cls(**kwargs)


class GenRecModelConfig(PretrainedConfig):
    """Base configuration class for generative recommendation models.

    This class extends the `PretrainedConfig` from the Hugging Face Transformers library
    to include common configuration parameters for generative recommendation models.
    """

    model_type = "genrec"

    def __init__(
        self,
        vocab_size: int = 1024,
        hidden_size: int = 256,
        is_encoder_decoder: bool = True,
        decoder_start_token_id: int = 0,
        pad_token_id: int = 0,
        tie_word_embeddings: bool = True,
        use_cache: bool = True,
        **kwargs,
    ) -> None:
        """Initializes the configuration with model hyperparameters.

        Args:
            vocab_size (int): Size of the Semantic ID vocabulary. Default is 1024.
            hidden_size (int): Dimensionality of the model's hidden representations. Default is 256.
            is_encoder_decoder (bool): Indicates if the model is an encoder-decoder architecture. Default is True.
            decoder_start_token_id (int): The token ID to start decoding with. Default is 0.
            pad_token_id (int): The token ID used for padding sequences. Default is 0.
            tie_word_embeddings (bool): Whether to tie the input and output word embeddings. Default is True.
            use_cache (bool): Whether the model should use past key values to speed up decoding. Default is True.
            **kwargs: Additional keyword arguments for the base `PretrainedConfig`.
        """
        super().__init__(
            is_encoder_decoder=is_encoder_decoder,
            decoder_start_token_id=decoder_start_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.use_cache = use_cache

        if "is_encoder_decoder" in kwargs:  # pragma: no cover - defensive check
            assert kwargs["is_encoder_decoder"] is True, "GenRecModel only supports encoder-decoder architectures."


class GenRecOutputFactory:  # pragma: no cover - factory class
    """Factory for creating `GenRecOutput` instances."""

    _registry: dict[str, Type[GenRecOutput]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Type[_GenRecOutput]], Type[_GenRecOutput]]:
        """Decorator to register a `GenRecOutput` implementation."""

        def decorator(output_cls: Type[_GenRecOutput]) -> Type[_GenRecOutput]:
            if name in cls._registry:
                raise ValueError(f"GenRec output '{name}' is already registered.")
            cls._registry[name] = output_cls
            return output_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs) -> GenRecOutput:
        """Creates an instance of a registered `GenRecOutput`."""
        if name not in cls._registry:
            raise ValueError(f"GenRec output '{name}' is not registered.")
        output_cls = cls._registry[name]
        return output_cls(**kwargs)


@dataclass
class GenRecOutput(Seq2SeqLMOutput):
    """Output class for encoder-decoder generative recommendation models.

    This class extends `Seq2SeqLMOutput` from the Hugging Face Transformers library
    and includes additional attributes.

    Args:
        model_loss (Optional[Float[torch.Tensor, ""]]): The computed model-specific
            loss value, if applicable. Note that the model-agnostic loss (e.g., CE loss)
            is handled outside of this class.
    """

    model_loss: Optional[Float[torch.Tensor, ""]] = None


class ShiftRightMixin(Generic[_GenRecModelConfig]):
    """Mixin class providing the `_shift_right` utility method for shifting input IDs
    to the right for decoder input sequences.
    """

    config: _GenRecModelConfig

    def _shift_right(self, input_ids: Int[torch.Tensor, "B L"]) -> Int[torch.Tensor, "B L"]:
        """Shifts input IDs one position to the right, prepending the decoder start token.
        This is used to create decoder input sequences.

        Args:
            input_ids (Int[torch.Tensor, "B L"]): Input token sequences of shape (batch_size, seq_len).

        Returns:
            Int[torch.Tensor, "B L"]: Shifted input token sequences.
        """
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        assert decoder_start_token_id is not None, "decoder_start_token_id must be defined in GenRecModelConfig."
        assert pad_token_id is not None, "pad_token_id must be defined in GenRecModelConfig."

        # shift input ids right by one position, prepending decoder_start_token_id
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id

        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids


class GenRecModelFactory:  # pragma: no cover - factory class
    """Factory for creating `GenRecModel` instances."""

    _registry: dict[str, Type[GenRecModel[Any, Any, Any]]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Type[_GenRecModel]], Type[_GenRecModel]]:
        """Decorator to register a `GenRecModel` implementation."""

        def decorator(model_cls: Type[_GenRecModel]) -> Type[_GenRecModel]:
            if name in cls._registry:
                raise ValueError(f"GenRec model '{name}' is already registered.")
            cls._registry[name] = model_cls
            return model_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs) -> GenRecModel[Any, Any, Any]:
        """Creates an instance of a registered `GenRecModel`."""
        if name not in cls._registry:
            raise ValueError(f"GenRec model '{name}' is not registered.")
        model_cls = cls._registry[name]
        return model_cls(**kwargs)

    @classmethod
    def from_pretrained(cls, name: str, path: Union[str, os.PathLike], **kwargs) -> GenRecModel[Any, Any, Any]:
        """Loads a pretrained instance of a registered `GenRecModel`."""
        if name not in cls._registry:
            raise ValueError(f"GenRec model '{name}' is not registered.")
        model_cls = cls._registry[name]
        return model_cls.from_pretrained(path, **kwargs)


class GenRecModel(
    PreTrainedModel,
    ShiftRightMixin[_GenRecModelConfig],
    Generic[_GenRecModelConfig, _GenRecOutput, _GenRecEncoderDecoderOutput],
    GenerationMixin,
    ABC,
):
    """Base class for encoder-decoder generative recommendation models that support beam
    search generation.

    This class extends the `PreTrainedModel` from the Hugging Face Transformers library
    and serves as a base for implementing specific generative recommendation models. This
    class also provides utilities for constrained generation (e.g., constrained beam search).

    Subclasses must specify the `config_class` attribute and implement the `forward` method.
    """

    config_class: Type[_GenRecModelConfig]
    output_class: Type[_GenRecOutput]
    main_input_name: str = "input_ids"  # main input for generation, i.e., SIDs
    supports_gradient_checkpointing = False  # change to True if implemented in subclass

    def __init__(
        self,
        config: _GenRecModelConfig,
    ) -> None:
        """Initializes the generative recommendation model.

        Args:
            config (_GenRecModelConfig): Configuration containing model hyperparameters.
        """
        super().__init__(config)
        self.config: _GenRecModelConfig
        assert self.config.is_encoder_decoder, "GenRecModel only supports encoder-decoder architectures."

        # By default, we assume encoder and decoder share the same token embeddings
        self.shared = nn.Embedding(config.vocab_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    @property
    @abstractmethod
    def encoder(self) -> nn.Module:  # pragma: no cover - abstract method
        """Returns the encoder module. You should implement this in subclasses."""
        pass

    @property
    @abstractmethod
    def decoder(self) -> nn.Module:  # pragma: no cover - abstract method
        """Returns the decoder module. You should implement this in subclasses."""
        pass

    def get_input_embeddings(self) -> nn.Module:
        """Returns the input embedding module."""
        return self.shared

    def set_input_embeddings(self, value: nn.Module) -> None:
        """Sets the input embedding module."""
        self.shared = value
        assert hasattr(self.encoder, "set_input_embeddings"), "Encoder does not support setting input embeddings."
        assert hasattr(self.decoder, "set_input_embeddings"), "Decoder does not support setting input embeddings."
        self.encoder.set_input_embeddings(value)
        self.decoder.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module:
        """Returns the output (LM head) projection."""
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        """Sets the output (LM head) projection."""
        self.lm_head = new_embeddings

    def get_encoder(self) -> nn.Module:  # pragma: no cover - method in abstract base class
        """Returns the encoder module."""
        return self.encoder

    def forward(  # pragma: no cover - method in abstract base class
        self,
        input_ids: Optional[Int[torch.Tensor, "B L_enc"]] = None,
        attention_mask: Optional[Int[torch.Tensor, "B L_enc"]] = None,
        decoder_input_ids: Optional[Int[torch.Tensor, "B L_dec"]] = None,
        decoder_attention_mask: Optional[Int[torch.Tensor, "B L_dec"]] = None,
        encoder_outputs: Optional[_GenRecEncoderDecoderOutput] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[Float[torch.Tensor, "B L_enc d"]] = None,
        decoder_inputs_embeds: Optional[Float[torch.Tensor, "B L_dec d"]] = None,
        labels: Optional[Int[torch.Tensor, "B L_dec"]] = None,
        cache_position: Optional[Int[torch.Tensor, "#L_dec"]] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_model_loss: Optional[bool] = None,
        **kwargs: Any,
    ) -> _GenRecOutput:
        """Defines the forward pass of the generative recommendation model.

        This method provides a typical interface for encoder-decoder models. You may override this method in
        subclasses to implement specific model architectures.

        Args:
            input_ids (Optional[Int[torch.Tensor, "B L_enc"]]): Input token sequences of shape (batch_size, seq_len).
            attention_mask (Optional[Int[torch.Tensor, "B L_enc"]]): Attention masks for inputs of shape
                (batch_size, seq_len).
            decoder_input_ids (Optional[Int[torch.Tensor, "B L_dec"]]): Decoder input token sequences
                of shape (batch_size, dec_seq_len). If `past_key_values` is used, only the last token
                of `decoder_input_ids` have to be input. Default is None.
            decoder_attention_mask (Optional[Int[torch.Tensor, "B L_dec"]]): Attention masks for decoder inputs
                of shape (batch_size, dec_seq_len). Default is None.
            encoder_outputs (Optional[_GenRecEncoderOutput]): Precomputed encoder outputs.
                This should be a subclass of `_GenRecEncoderOutput`. Default is None.
            past_key_values (Optional[Cache]): Cached key and value tensors for faster decoding. Default is None.
            inputs_embeds (Optional[Float[torch.Tensor, "B L d"]]): Input embeddings of `input_ids` of shape
                (batch_size, seq_len, hidden_size). If provided, `input_ids` will be ignored. Default is None.
            decoder_inputs_embeds (Optional[Float[torch.Tensor, "B L_dec d"]]): Input embeddings of
                `decoder_input_ids` of shape (batch_size, dec_seq_len, hidden_size). If provided,
                `decoder_input_ids` will be ignored. Default is None.
            labels (Optional[Int[torch.Tensor, "B L_dec"]]): Target token sequences for computin the loss, of
                shape (batch_size, dec_seq_len). Default is None.
            cache_position (Optional[Int[torch.Tensor, "#L_dec"]]): Positions for caching in the decoder.
                Default is None.
            use_cache (Optional[bool]): Whether to use past key values to speed up decoding. Default is None.
            output_attentions (Optional[bool]): Whether to return attention weights. Default is None.
            output_hidden_states (Optional[bool]): Whether to return hidden states. Default is None.
            output_model_loss (Optional[bool]): Whether to compute and return the model-specific loss.
                Default is None.
            **kwargs (Any): Additional keyword arguments for the model.

        Returns:
            _GenRecOutput: Model outputs packaged as a `GenRecOutput` object.
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if use_cache and self.training:
            use_cache = False  # disable use_cache during training to ensure consistent behavior

        # Encode if needed (training, first stage of generation)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
        assert encoder_outputs is not None
        hidden_states = encoder_outputs.last_hidden_state

        # Decoding
        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = self._shift_right(labels)

        decoder_outputs: _GenRecEncoderDecoderOutput = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = decoder_outputs.last_hidden_state

        # Compute logits
        # NOTE: Some models, e.g., T5, scale the logits by d_model ** -0.5 before the lm_head
        # Here we assume that the decoder will apply a final layernorm and thus no scaling is needed
        logits = self.lm_head(sequence_output)

        # NOTE: We do not compute the model-agnostic loss (e.g., CE loss) , compute it in `GenRecTrainer` instead
        # By default, we set model_loss to None, if you have a model-specific loss, compute it and set it here
        model_loss = torch.tensor(0.0, device=logits.device)

        return self.output_class(
            loss=None,  # model-agnostic loss to be computed outside
            logits=logits,
            past_key_values=decoder_outputs.past_key_values,  # type: ignore - EncoderDecoderCache
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            model_loss=model_loss if output_model_loss else None,
        )
