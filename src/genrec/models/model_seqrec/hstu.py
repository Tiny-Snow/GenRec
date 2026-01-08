"""SeqRec Model: HSTU."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

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
        add_ffn: bool = False,
        softmax_attention: bool = False,
        attention_norm: bool = False,
        time_interval: float = 1.0,
        relative_position_bias: bool = True,
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
            add_ffn (bool): Whether to add feed-forward network after attention in each HSTU layer.
                default is False.
            softmax_attention (bool): Whether to use softmax-based attention mechanism rather than
                the original silu-based attention mechanism. Default is False.
            attention_norm (bool): Whether to apply row-wise normalization to attention scores.
                Default is False.
            time_interval (float): Factor to divide Unix timestamps by before bucketization. Default is 1.0
                (seconds). Use larger values (e.g., 86400) to operate on coarser units such as days.
            relative_position_bias (bool): Whether to use relative position bias. Default is True.
            **kwargs (Any): Additional keyword arguments for the base `SeqRecModelConfig`.
        """
        super().__init__(**kwargs)
        self.max_seq_len = max_seq_len
        self.num_buckets = num_buckets
        self.linear_dropout = linear_dropout
        self.attention_dropout = attention_dropout
        self.final_layer_norm = final_layer_norm
        self.add_ffn = add_ffn
        self.softmax_attention = softmax_attention
        self.attention_norm = attention_norm
        self.time_interval = time_interval
        self.relative_position_bias = relative_position_bias


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
    - All the layer normalizations are implemented using `RMSNorm`. We also provide an option
        to apply a final layer normalization after the last HSTU layer.
    - Changes the industrial implementation of relative position and time attention bias
        to a more readable `RelativeBucketedTimeAndPositionAttentionBias`.
    - Provide an option to add feed-forward network after attention in each HSTU layer.
    - Provide an option to apply softmax-based attention rather than the original
        silu-based attention mechanism, as well as an option to apply row-wise normalization
        to attention scores before applying softmax, which may be more effective in some cases.
    - Provide an option to enable/disable relative position bias. If enabled, the relative
        position bias is added together with attention scores based on time intervals and
        position differences. We also allow the user to adjust the granularity of time intervals
        by setting the `time_interval` parameter in the model configuration.

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
                    add_ffn=config.add_ffn,
                    softmax_attention=config.softmax_attention,
                    attention_norm=config.attention_norm,
                    time_interval=config.time_interval,
                    relative_position_bias=config.relative_position_bias,
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
            HSTUModelOutput: Model outputs packaged as a `HSTUModelOutput` descendant.
        """

        timestamps: Optional[Int[torch.Tensor, "B L"]] = kwargs.pop("timestamps", None)
        if timestamps is None:  # pragma: no cover - defensive check
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
