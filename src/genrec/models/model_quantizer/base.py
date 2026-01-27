"""Base model for quantizer."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Generic, Optional, Sequence, Type, TypeVar, Union

from jaxtyping import Float, Int
import torch
from transformers import PretrainedConfig, PreTrainedModel
from transformers.utils.generic import ModelOutput

__all__ = [
    "QuantizerModel",
    "QuantizerModelFactory",
    "QuantizerModelConfig",
    "QuantizerModelConfigFactory",
    "QuantizerOutput",
    "QuantizerOutputFactory",
]

_QuantizerModelConfig = TypeVar("_QuantizerModelConfig", bound="QuantizerModelConfig")
_QuantizerOutput = TypeVar("_QuantizerOutput", bound="QuantizerOutput")
_QuantizerModel = TypeVar("_QuantizerModel", bound="QuantizerModel[Any, Any]")


class QuantizerModelConfigFactory:  # pragma: no cover - factory class
    """Factory for creating `QuantizerModelConfig` instances."""

    _registry: dict[str, Type[QuantizerModelConfig]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Type[_QuantizerModelConfig]], Type[_QuantizerModelConfig]]:
        """Decorator to register a `QuantizerModelConfig` implementation."""

        def decorator(config_cls: Type[_QuantizerModelConfig]) -> Type[_QuantizerModelConfig]:
            if name in cls._registry:
                raise ValueError(f"Quantizer model config '{name}' is already registered.")
            cls._registry[name] = config_cls
            return config_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs) -> QuantizerModelConfig:
        """Creates an instance of a registered `QuantizerModelConfig`."""
        if name not in cls._registry:
            raise ValueError(f"Quantizer model config '{name}' is not registered.")
        config_cls = cls._registry[name]
        return config_cls(**kwargs)


class QuantizerModelConfig(PretrainedConfig):
    """Base configuration class for quantizer models.

    This class extends the `PretrainedConfig` from the Hugging Face Transformers library
    and serves as a base for implementing specific quantizer model configurations.

    Subclasses must specify the `model_type` attribute.
    """

    model_type = "quantizer"

    def __init__(
        self,
        embed_dim: int = 768,
        hidden_sizes: Sequence[int] = (512, 256, 128),
        num_codebooks: int = 3,
        codebook_size: int = 256,
        codebook_dim: int = 32,
        **kwargs,
    ) -> None:
        """Initializes the configuration with model hyperparameters.

        Args:
            embed_dim (int): Dimensionality of the input dense embeddings.
            hidden_sizes (Sequence[int]): Sizes of hidden layers in the quantizer encoder. Note
                that a Linear(hidden_sizes[-1], embed_dim) layer is appended to the encoder.
                The decoder will be symmetric to the encoder. Default is (512, 256, 128).
            num_codebooks (int): Number of codebooks in the quantizer. Default is 3.
            codebook_size (int): Number of codes in each codebook. Default is 256.
            codebook_dim (int): Dimensionality of each code in the codebooks. Default is 32.
            **kwargs (Any): Additional keyword arguments for the base `PretrainedConfig`.
        """
        super().__init__(**kwargs)

        self.embed_dim = embed_dim
        self.hidden_sizes = hidden_sizes
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim


class QuantizerOutputFactory:  # pragma: no cover - factory class
    """Factory for creating `QuantizerOutput` instances."""

    _registry: dict[str, Type[QuantizerOutput]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Type[_QuantizerOutput]], Type[_QuantizerOutput]]:
        """Decorator to register a `QuantizerOutput` implementation."""

        def decorator(output_cls: Type[_QuantizerOutput]) -> Type[_QuantizerOutput]:
            if name in cls._registry:
                raise ValueError(f"Quantizer output '{name}' is already registered.")
            cls._registry[name] = output_cls
            return output_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs) -> QuantizerOutput:
        """Creates an instance of a registered `QuantizerOutput`."""
        if name not in cls._registry:
            raise ValueError(f"Quantizer output '{name}' is not registered.")
        output_cls = cls._registry[name]
        return output_cls(**kwargs)


@dataclass
class QuantizerOutput(ModelOutput):
    """Base output class for quantizer models.

    Attributes:
        semantic_ids (Int[torch.Tensor, "B C"]): Semantic IDs assigned by the quantizer,
            i.e., the indices of the selected codes from each codebook. The shape is
            (batch_size, num_codebooks).
        quantized_embeddings (Optional[Float[torch.Tensor, "B C D_c"]]): Quantized
            embeddings corresponding to the semantic IDs, if applicable. The shape is
            (batch_size, num_codebooks, codebook_dim).
        residual_embeddings (Optional[Float[torch.Tensor, "B C D_c"]]): Residual embeddings
            before quantization to the corresponding codebooks, if applicable. The shape is
            (batch_size, num_codebooks, codebook_dim).
        decoded_embeddings (Optional[Float[torch.Tensor, "B D"]]): Reconstructed dense
            embeddings from the quantizer, if applicable. The shape is (batch_size, embed_dim).
        reconstruction_loss (Optional[Float[torch.Tensor, "B"]]): The reconstruction loss
            value, if applicable.
        codebook_loss (Optional[Float[torch.Tensor, "B"]]): The codebook loss value,
            if applicable.
        commitment_loss (Optional[Float[torch.Tensor, "B"]]): The commitment loss value,
            if applicable.
        model_loss (Optional[Float[torch.Tensor, ""]]): The computed model-specific
            loss value, if applicable. Note that the model-agnostic loss (e.g.,
            reconstruction or commitment losses) is handled outside of this class.

    .. note::
        In typical quantizers, e.g., RQ-VAE, the `residual_embeddings` are supposed to be
        close to the `quantized_embeddings` by optimizing the commitment loss. The
        `decoded_embeddings` are supposed to be close to the original dense embeddings
        by optimizing the reconstruction loss.

    .. note::
        The STE (Straight-Through Estimator) trick is applied for `quantized_embeddings`
        to allow gradient backpropagation during training. That is, the forward pass uses
        the quantized embeddings, while in the backward pass, the gradients are directly
        passed to the input embeddings before quantization.

    .. note::
        The output `semantic_ids`, e.g., `<A_0>, <B_23>, <C_5>`, may not be completely
        unique for different input embeddings due to collisions in the quantization process.
        In practice, some strategies such as adding an addition anti-collision code, e.g.,
        `<Z_0>`, can be employed to mitigate this issue. In addition, each code, e.g., `<B_23>`,
        is originally ranged from 0 to `codebook_size - 1` within each codebook. To convert
        them to global unique IDs, an offset can be added based on the codebook index, e.g.,
        `B_23` -> `B_23 + codebook_size * 1 + 1` (The last `+1` is for reserving the padding ID).
        These logics are expected to be handled in `QuantizerModel.post_process_quantized_ids`.
    """

    semantic_ids: Int[torch.Tensor, "B C"]
    quantized_embeddings: Optional[Float[torch.Tensor, "B C D_c"]] = None
    residual_embeddings: Optional[Float[torch.Tensor, "B C D_c"]] = None
    decoded_embeddings: Optional[Float[torch.Tensor, "B D"]] = None
    reconstruction_loss: Optional[Float[torch.Tensor, "B"]] = None
    codebook_loss: Optional[Float[torch.Tensor, "B"]] = None
    commitment_loss: Optional[Float[torch.Tensor, "B"]] = None
    model_loss: Optional[Float[torch.Tensor, ""]] = None


class QuantizerModelFactory:  # pragma: no cover - factory class
    """Factory for creating `QuantizerModel` instances."""

    _registry: dict[str, Type[QuantizerModel[Any, Any]]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Type[_QuantizerModel]], Type[_QuantizerModel]]:
        """Decorator to register a `QuantizerModel` implementation."""

        def decorator(model_cls: Type[_QuantizerModel]) -> Type[_QuantizerModel]:
            if name in cls._registry:
                raise ValueError(f"Quantizer model '{name}' is already registered.")
            cls._registry[name] = model_cls
            return model_cls

        return decorator

    @classmethod
    def create(cls, name: str, config: QuantizerModelConfig) -> QuantizerModel[Any, Any]:
        """Creates an instance of a registered `QuantizerModel`."""
        if name not in cls._registry:
            raise ValueError(f"Quantizer model '{name}' is not registered.")
        model_cls = cls._registry[name]
        return model_cls(config)

    @classmethod
    def from_pretrained(cls, name: str, path: Union[str, os.PathLike], **kwargs) -> QuantizerModel[Any, Any]:
        """Loads a pretrained instance of a registered `QuantizerModel`."""
        if name not in cls._registry:
            raise ValueError(f"Quantizer model '{name}' is not registered.")
        model_cls = cls._registry[name]
        return model_cls.from_pretrained(path, **kwargs)


class QuantizerModel(PreTrainedModel, Generic[_QuantizerModelConfig, _QuantizerOutput], ABC):
    """Base class for quantizer models.

    This class extends the `PreTrainedModel` from the Hugging Face Transformers library
    and serves as a base for implementing specific quantizer models.

    Subclasses must specify the `config_class` attribute and implement the `forward` and
    `initialize_codebooks` methods. We provide a default implementation for
    `post_process_quantized_ids`, which can be overridden if needed.
    """

    config_class: Type[_QuantizerModelConfig]

    def __init__(self, config: _QuantizerModelConfig) -> None:
        """Initializes the quantizer model.

        Args:
            config (_QuantizerModelConfig): Configuration containing model hyperparameters.
        """
        super().__init__(config)
        self.config: _QuantizerModelConfig

    @abstractmethod
    def forward(  # pragma: no cover - abstract method
        self,
        item_id: Int[torch.Tensor, "B"],
        item_embedding: Float[torch.Tensor, "B D"],
        output_loss: bool = False,
        output_model_loss: bool = False,
        output_embeddings: bool = False,
        **kwargs,
    ) -> _QuantizerOutput:
        """Performs a forward pass through the quantizer model.

        Args:
            item_id (Int[torch.Tensor, "B"]): Item IDs corresponding to the input embeddings.
            item_embedding (Float[torch.Tensor, "B D"]): Dense item embeddings to be quantized.
            output_loss (bool): Whether to compute and return the reconstruction and commitment losses. Default is False.
            output_model_loss (bool): Whether to compute and return the model-specific loss. Default is False.
            output_embeddings (bool): Whether to return the (quantized, residual, and decoded) embeddings. Default is False.
            **kwargs (Any): Additional keyword arguments for the forward pass.

        Returns:
            _QuantizerOutput: Model outputs packaged as a `QuantizerOutput` instance.
        """
        ...

    @abstractmethod
    def initialize_codebooks(  # pragma: no cover - abstract method
        self,
        item_embeddings: Float[torch.Tensor, "I D"],
        **kwargs,
    ) -> None:
        """Initializes the codebooks using the provided item embeddings.

        This method is typically called when a specific global initialization strategy
        is desired, e.g., when the `kmeans_init` is set to True in the RQ-VAE model
        configuration. In this case, it performs k-means clustering on the item embeddings
        to initialize the codebooks. You may override this method in subclasses to implement
        custom initialization logic.

        Args:
            item_embeddings (Float[torch.Tensor, "I D"]): Dense item embeddings used for
                initializing the codebooks.
        """
        ...

    def post_process_quantized_ids(
        self,
        semantic_ids: Int[torch.Tensor, "B C"],
    ) -> Int[torch.Tensor, "B C_new"]:
        """Post-processes the semantic IDs to ensure global uniqueness.

        This method converts the local semantic IDs (ranging from 0 to `codebook_size - 1`
        within each codebook) to globally unique IDs by adding an offset based on the
        codebook index. Additionally, it handles anti-collision codes by appending an
        extra code `Z_0` to the codebooks (so `C_new = C + 1` by default). Note that the
        final codes are shifted by 1 to reserve the padding ID zero (which corresponds to
        `<A_0>, <B_0>, <C_0>, ..., <Z_0>`). You may override this method in subclasses to
        implement custom post-processing logic.

        Args:
            semantic_ids (Int[torch.Tensor, "B C"]): Local semantic IDs assigned by the quantizer.
                The shape is (batch_size, num_codebooks), where the `batch_size` is expected to be
                the item number in most cases.

        Returns:
            Int[torch.Tensor, "B C_new"]: Globally unique semantic IDs after post-processing.
                In this implementation, `C_new = C + 1` to account for the anti-collision codes.
        """
        B, C = semantic_ids.shape
        assert C == self.config.num_codebooks, "Unexpected number of codebooks in semantic IDs."

        # append anti-collision code at the end of each codebook
        anti_collision_codes = torch.zeros((B, 1), dtype=semantic_ids.dtype, device=semantic_ids.device)
        seen_counts = {}
        for i in range(B):
            key = tuple(semantic_ids[i].tolist())
            count = seen_counts.get(key, 0)
            anti_collision_codes[i, 0] = count
            seen_counts[key] = count + 1
        semantic_ids = torch.cat([semantic_ids, anti_collision_codes], dim=1)

        # add offset based on codebook index
        for codebook_idx in range(semantic_ids.shape[1]):
            offset = codebook_idx * self.config.codebook_size + 1  # +1 for reserving padding ID
            semantic_ids[:, codebook_idx] += offset

        return semantic_ids
