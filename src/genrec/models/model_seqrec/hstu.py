"""SeqRec Model: HSTU."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from jaxtyping import Bool, Float, Int
import torch
import torch.nn as nn

from ..modules import LearnableInputPositionalEmbedding, RMSNorm, SequentialTransductionUnit, create_attention_mask
from .base import (
    SeqRecModel,
    SeqRecModelConfig,
    SeqRecModelConfigFactory,
    SeqRecModelFactory,
    SeqRecOutput,
    SeqRecOutputFactory,
)

__all__ = [
    "HSTUModel",
    "HSTUModelConfig",
    "HSTUModelOutput",
]


@SeqRecModelConfigFactory.register("hstu")
class HSTUModelConfig(SeqRecModelConfig):
    """Configuration class for HSTU model, which extends the base `SeqRecModelConfig`."""

    def __init__(
        self,
        max_seq_len: int = 512,
        num_buckets: int = 128,
        linear_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        final_layer_norm: bool = False,
        **kwargs,
    ) -> None:
        """Initializes the configuration with model hyperparameters.

        Args:
            max_seq_len (int): Maximum sequence length for the model. It is used for configuring
                learnable positional embeddings. Default is 512.
            num_buckets (int): Number of buckets for relative position and time embeddings.
                Default is 128.
            linear_dropout (float): Dropout rate for input embeddings and GLU. Default is 0.0.
            attention_dropout (float): Dropout rate for attention weights. Default is 0.0.
            final_layer_norm (bool): Whether to apply a final layer normalization. Default is False.
            **kwargs (Any): Additional keyword arguments for the base `SeqRecModelConfig`.
        """
        super().__init__(**kwargs)
        self.max_seq_len = max_seq_len
        self.num_buckets = num_buckets
        self.linear_dropout = linear_dropout
        self.attention_dropout = attention_dropout
        self.final_layer_norm = final_layer_norm


@SeqRecOutputFactory.register("hstu")
@dataclass
class HSTUModelOutput(SeqRecOutput):
    """Output class for HSTU model.

    The `HSTUModelOutput` extends the base `SeqRecOutput` without adding any additional attributes.
    """

    pass


@SeqRecModelFactory.register("hstu")
class HSTUModel(SeqRecModel[HSTUModelConfig, HSTUModelOutput]):
    """HSTU model implementation.

    Here we implement HSTU (Hierarchical Sequential Transformer Unit) based on its
    official code (https://github.com/meta-recsys/generative-recommenders). Some
    slight modifications are made as follows:

    - The input positional embeddings are implemented using `LearnableInputPositionalEmbedding`,
        which directly adds learnable positional embeddings to the input embeddings
        without scaling with sqrt(hidden_size).
    - All the layer normalizations are implemented using `RMSNorm`.
    - Changes the industrial implementation of relative position and time attention bias
        to a more readable `RelativeBucketedTimeAndPositionAttentionBias`.

    .. note::
        It suggests setting `linear_dropout` to 0.0 to stabilize training, as we observed
        several gradient explosion issues when using non-zero linear dropout rates. Even
        only the input embedding dropout is set to non-zero, the training could still
        be unstable. However, `attention_dropout` can be set to a non-zero value without
        causing instability.

    References:
        - Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for
            Generative Recommendations. ICML '24.
    """

    config_class = HSTUModelConfig

    def __init__(self, config: HSTUModelConfig) -> None:
        """Initializes HSTU model with the given configuration."""
        super().__init__(config)
        self.config: HSTUModelConfig

        assert (
            config.hidden_size % config.num_attention_heads == 0
        ), "hidden_size must be divisible by num_attention_heads."
        self.head_dim = config.hidden_size // config.num_attention_heads

        # NOTE: in HSTU, the dropout_rate of input embeddings and GLU are both set to linear_dropout.
        self.input_pos_emb = LearnableInputPositionalEmbedding(
            max_position_embeddings=config.max_seq_len,
            embed_dim=config.hidden_size,
            dropout_rate=config.linear_dropout,
        )

        self.layers = nn.ModuleList(
            [
                SequentialTransductionUnit(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_attention_heads,
                    max_seq_len=config.max_seq_len,
                    num_buckets=config.num_buckets,
                    linear_dropout=config.linear_dropout,
                    attention_dropout=config.attention_dropout,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )

        if config.final_layer_norm:
            self.final_layer_norm = RMSNorm(config.hidden_size)
        else:
            self.final_layer_norm = nn.Identity()

        self.gradient_checkpointing = False  # disable gradient checkpointing by default
        self.post_init()  # use PretrainedModel's default weight initialization

    def forward(
        self,
        input_ids: Int[torch.Tensor, "B L"],
        attention_mask: Int[torch.Tensor, "B L"],
        output_model_loss: bool = False,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        **kwargs,
    ) -> HSTUModelOutput:
        """Forward pass for HSTU model.

        Args:
            input_ids (Int[torch.Tensor, "B L"]): Input item ID sequences of shape (batch_size, seq_len).
            attention_mask (Int[torch.Tensor, "B L"]): Attention masks of shape (batch_size, seq_len).
            output_model_loss (bool): Whether to compute and return the model-specific loss. Default is False.
            output_hidden_states (bool): Whether to return hidden states from all layers. Default is False.
            output_attentions (bool): Whether to return attention weights from all layers. Default is False.
            **kwargs (Any): Additional keyword arguments for the model.

        Keywords Args:
            timestamps (Int[torch.Tensor, "B L"]): Timestamps corresponding to each item in the input sequences.
                The timestamps are assumed to be Unix format (in seconds).

        Returns:
            HSTUModelOutput: Model outputs packaged as a `SeqRecOutput` descendant.
        """

        timestamps: Optional[Int[torch.Tensor, "B L"]] = kwargs.pop("timestamps", None)
        if timestamps is None:
            raise ValueError("HSTUModel.forward requires `timestamps` to be provided via kwargs.")

        hidden_states: Float[torch.Tensor, "B L d"]
        hidden_states = self.embed_tokens(input_ids)

        causal_mask: Bool[torch.Tensor, "B 1 L L"]
        causal_mask = create_attention_mask(attention_mask, is_causal=True, mask_value=1).bool()
        causal_mask = ~causal_mask  # True for valid positions, False for masked positions.

        model_loss = None  # By default, HSTU does not compute model loss internally.
        all_hidden_states: List[Float[torch.Tensor, "B L d"]] = []
        all_attentions: List[Float[torch.Tensor, "B H L L"]] = []

        hidden_states = self.input_pos_emb(hidden_states)

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            hidden_states, attn_weights = layer(
                hidden_states,
                attention_mask=causal_mask,
                timestamps=timestamps,
            )

            if output_attentions:
                all_attentions.append(attn_weights)

        # NOTE: in the official HSTU code, there is no final layer norm,
        # as it apply L2 normalization after getting the final item representations.
        # Here we still keep an optional final layer norm for other loss functions' sake.
        hidden_states = self.final_layer_norm(hidden_states)

        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        return HSTUModelOutput(
            last_hidden_state=hidden_states,
            model_loss=model_loss if output_model_loss else None,
            hidden_states=tuple(all_hidden_states) if output_hidden_states else None,
            attentions=tuple(all_attentions) if output_attentions else None,
        )
