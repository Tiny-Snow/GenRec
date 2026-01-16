"""SeqRec Model: HSTU."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from jaxtyping import Float, Int
import torch
import torch.nn as nn

from ..modules import (
    LearnableInputPositionalEmbedding,
    RMSNorm,
    RotaryEmbedding,
    SequentialTransductionUnit,
    create_attention_mask,
)
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
        linear_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        max_seq_len: int = 512,
        num_buckets: int = 128,
        enable_input_pos_emb: bool = True,
        enable_learnable_rel_posemb: bool = True,
        enable_attention_gating: bool = True,
        enable_ffn: bool = False,
        enable_final_layernorm: bool = False,
        **kwargs,
    ) -> None:
        """Initializes the configuration with model hyperparameters.

        Args:
            linear_dropout (float): Dropout rate for input embeddings and attention output before
                the final output projection, which is applied in `av_output` when `attention_gating`
                is enabled. Default is 0.0.
            attention_dropout (float): Dropout rate for attention weights. Default is 0.0.
            max_seq_len (int): Maximum sequence length for relative positional embeddings. Default is 512.
            num_buckets (int): Number of buckets for relative positional embeddings. Default is 128.
            enable_input_pos_emb (bool): Whether to use learnable input positional embeddings. Default is True.
            enable_learnable_rel_posemb (bool): Whether to use learnable relative positional embeddings.
                If False, RoPE will be used instead. Default is True.
            enable_attention_gating (bool): Whether to enable the attention gating mechanism. If False,
                standard attention computation is used. Default is True.
            enable_ffn (bool): Whether to include a feed-forward network after attention. Default is False.
            enable_final_layernorm (bool): Whether to apply a final layer normalization after the last HSTU layer.
                This is recommended when using dot-product-based loss functions. Default is False.
            **kwargs (Any): Additional keyword arguments for the base `SeqRecModelConfig`.
        """
        super().__init__(**kwargs)
        self.linear_dropout = linear_dropout
        self.attention_dropout = attention_dropout
        self.max_seq_len = max_seq_len
        self.num_buckets = num_buckets
        self.enable_input_pos_emb = enable_input_pos_emb
        self.enable_learnable_rel_posemb = enable_learnable_rel_posemb
        self.enable_attention_gating = enable_attention_gating
        self.enable_ffn = enable_ffn
        self.enable_final_layernorm = enable_final_layernorm


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

    Here we implement HSTU (Hierarchical Sequential Transformer Unit) based on the Meta's
    official code (https://github.com/meta-recsys/generative-recommenders). We generalize
    the original implementation to allow more flexibility in model configuration and boost
    model performance. The main differences are summarized as follows:

    - The input positional embeddings are implemented using `LearnableInputPositionalEmbedding`,
        which directly adds learnable positional embeddings to the input embeddings without
        scaling with sqrt(hidden_size). We also provide an option `enable_input_pos_emb` to disable
        input positional embeddings.
    - All the layer normalizations are implemented using `RMSNorm`. We also provide an option
        `enable_final_layernorm` to apply a final layer normalization after the last HSTU layer.
    - Changes the industrial implementation of relative position and time attention bias
        to a more readable `RelativeBucketedTimeAndPositionAttentionBias`.
    - Provide an option `enable_learnable_rel_posemb` to switch the original learnable relative
        positional embeddings with Rotary Positional Embeddings (RoPE) for better extrapolation
        to longer sequences and improved performance.
    - Provide an option `enable_attention_gating` to disable the original attention gating mechanism,
        allowing for a standard attention computation, which can be beneficial in certain scenarios
        where gating may not be stable.
    - Provide an option `enable_ffn` to restore the FFN after attention, which was removed in the
        original STU design, to enhance the model's capacity.

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
        self.input_pos_emb: Optional[LearnableInputPositionalEmbedding] = None
        if config.enable_input_pos_emb:
            self.input_pos_emb = LearnableInputPositionalEmbedding(
                max_position_embeddings=config.max_seq_len,
                embed_dim=config.hidden_size,
                dropout_rate=config.linear_dropout,
            )

        self.rotary_emb: Optional[RotaryEmbedding] = None
        if not config.enable_learnable_rel_posemb:
            self.rotary_emb = RotaryEmbedding(head_dim=self.head_dim)

        self.layers = nn.ModuleList(
            [
                SequentialTransductionUnit(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_attention_heads,
                    intermediate_size=config.hidden_size * 4,
                    attention_dropout=config.attention_dropout,
                    linear_dropout=config.linear_dropout,
                    max_seq_len=config.max_seq_len,
                    num_buckets=config.num_buckets,
                    enable_learnable_rel_posemb=config.enable_learnable_rel_posemb,
                    enable_attention_gating=config.enable_attention_gating,
                    enable_ffn=config.enable_ffn,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )

        self.final_layernorm: Optional[RMSNorm] = None
        if config.enable_final_layernorm:
            self.final_layernorm = RMSNorm(config.hidden_size)

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

        causal_mask: Float[torch.Tensor, "B 1 L L"]
        causal_mask = create_attention_mask(attention_mask, is_causal=True)

        position_embeddings: Optional[
            Tuple[Float[torch.Tensor, "B L head_dim"], Float[torch.Tensor, "B L head_dim"]]
        ] = None
        if not self.config.enable_learnable_rel_posemb and self.rotary_emb is not None:
            position_embeddings = self.rotary_emb(hidden_states)

        model_loss = None  # By default, HSTU does not compute model loss internally.
        all_hidden_states: List[Float[torch.Tensor, "B L d"]] = []
        all_attentions: List[Float[torch.Tensor, "B H L L"]] = []

        if self.config.enable_input_pos_emb and self.input_pos_emb is not None:
            hidden_states = self.input_pos_emb(hidden_states)

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            hidden_states, attn_weights = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                timestamps=timestamps,
            )

            if output_attentions:
                all_attentions.append(attn_weights)

        # NOTE: in the official HSTU code, there is no final layer norm,
        # as it apply L2 normalization after getting the final item representations.
        # Here we still keep an optional final layer norm for other loss functions' sake.
        if self.config.enable_final_layernorm and self.final_layernorm is not None:
            hidden_states = self.final_layernorm(hidden_states)

        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        return HSTUModelOutput(
            last_hidden_state=hidden_states,
            model_loss=model_loss if output_model_loss else None,
            hidden_states=tuple(all_hidden_states) if output_hidden_states else None,
            attentions=tuple(all_attentions) if output_attentions else None,
        )
