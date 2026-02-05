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
    SeqRecModelConfigFactory,
    SeqRecModelFactory,
    SeqRecOutputFactory,
)

from .hstu import HSTUModelConfig, HSTUModelOutput

__all__ = [
    "HSTUMOJITOModel",
    "HSTUMOJITOModelConfig",
]


@SeqRecModelConfigFactory.register("hstu_mojito")
class HSTUMOJITOModelConfig(HSTUModelConfig):
    def __init__(
        self,
        lambda_trans_seq: float = 0.1,
        num_fism_items: int = 20,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.lambda_trans_seq = lambda_trans_seq
        self.num_fism_items = num_fism_items
        self.num_buckets = 20  # Number of coarse popularity buckets for MOJITO


@SeqRecOutputFactory.register("hstu_mojito")
@dataclass
class HSTUMOJITOModelOutput(HSTUModelOutput):
    """HSTU model output class for MOJITO."""

    pass


@SeqRecModelFactory.register("hstu_mojito")
class HSTUMOJITOModel(SeqRecModel[HSTUMOJITOModelConfig, HSTUModelOutput]):
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

    config_class = HSTUMOJITOModelConfig

    def __init__(self, config: HSTUMOJITOModelConfig) -> None:
        """Initializes HSTU model with the given configuration."""
        super().__init__(config)
        self.config: HSTUMOJITOModelConfig

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

        self.time_emb = MOJITOTimeEmbedding(num_buckets=config.num_buckets, time_emb_dim=config.hidden_size)
        self.time_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.temporal_mix = GaussianMixtureTemporalEmbedding(embedding_dim=config.hidden_size)

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
        time_embeddings: Float[torch.Tensor, "B L d"] = self.time_emb(timestamps)

        hidden_states: Float[torch.Tensor, "B L d"]
        hidden_states = self.temporal_mix(
            item_emb=self.embed_tokens(input_ids),
            time_emb=self.time_proj(time_embeddings),
        )
        causal_mask: Float[torch.Tensor, "B 1 L L"]
        causal_mask = create_attention_mask(attention_mask, is_causal=True)

        position_embeddings: Optional[
            Tuple[Float[torch.Tensor, "B L head_dim"], Float[torch.Tensor, "B L head_dim"]]
        ] = None
        if not self.config.enable_learnable_rel_posemb and self.rotary_emb is not None:
            position_embeddings = self.rotary_emb(hidden_states)

        if self.config.enable_input_pos_emb and self.input_pos_emb is not None:
            hidden_states = self.input_pos_emb(hidden_states)

        model_loss = None  # By default, HSTU does not compute model loss internally.
        all_hidden_states: List[Float[torch.Tensor, "B L d"]] = []
        all_attentions: List[Float[torch.Tensor, "B H L L"]] = []

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


class MOJITOTimeEmbedding(nn.Module):
    def __init__(self, num_buckets: int, time_emb_dim: int):
        super().__init__()
        self.num_buckets = num_buckets
        self.time_embedding = nn.Embedding(num_buckets + 1, time_emb_dim)

    @staticmethod
    def bucketize_timestamps(timestamps, num_buckets):
        x = timestamps.abs().clamp(min=1).float()
        buckets = (torch.log(x) / 0.301).long()
        buckets = torch.clamp(buckets, min=0, max=num_buckets)
        return buckets

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        buckets = self.bucketize_timestamps(timestamps, self.num_buckets)
        return self.time_embedding(buckets)


class GaussianMixtureTemporalEmbedding(nn.Module):
    """Gaussian Mixture-based Temporal Embedding"""

    def __init__(self, embedding_dim: int, init_noise_std: float = 0.02):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.register_buffer("time_emb_scale", torch.tensor(embedding_dim**0.5))

        # Learnable Gaussian mixture weights (initialize equally distributed)
        # self.temporal_noise = nn.Parameter(torch.ones(embedding_dim) * init_noise_std)

    def forward(self, item_emb: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            item_emb: [B, L, d] item embeddings from self.embed_tokens
            time_emb: [B, L, d] time embeddings from self.time_emb

        Returns:
            Combined embedding with Gaussian mixture weighting: [B, L, d]
        """
        # # Add temporal noise for uncertainty modeling
        # noise = torch.randn_like(time_emb) * self.temporal_noise.view(1, 1, -1)
        # noisy_time_emb = time_emb + noise

        # Combine item and temporal embeddings using Gaussian weights
        final_embedding = item_emb + time_emb / self.time_emb_scale
        return final_embedding
