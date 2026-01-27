"""SeqRec Model: SASRec."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from jaxtyping import Float, Int
import torch
import torch.nn as nn

from ..modules import RMSNorm, RotaryEmbedding, create_attention_mask, LlamaDecoderLayer
from .base import SeqRecModel, SeqRecModelConfigFactory, SeqRecModelFactory, SeqRecOutputFactory
from .sasrec import SASRecModelConfig, SASRecModelOutput

__all__ = [
    "SASRecTPABModel",
    "SASRecTPABModelConfig",
    "SASRecTPABModelOutput",
]


@SeqRecModelConfigFactory.register("sasrec_tpab")
class SASRecTPABModelConfig(SASRecModelConfig):
    """Configuration class for SASRec model, which extends the base `SASRecModelConfig`."""

    def __init__(
        self,
        num_buckets: int = 20,
        **kwargs,
    ) -> None:
        """Initializes the configuration with model hyperparameters.

        Args:
            num_buckets (int): Number of coarse popularity buckets for TPAB. Default is 20.
            **kwargs (Any): Additional keyword arguments for the base `SeqRecModelConfig`.
        """
        super().__init__(**kwargs)


@SeqRecOutputFactory.register("sasrec_tpab")
@dataclass
class SASRecTPABModelOutput(SASRecModelOutput):
    """Output class for SASRec model.

    The `SASRecTPABModelOutput` extends the base `SASRecModelOutput` without adding any additional attributes.
    """

    pass


@SeqRecModelFactory.register("sasrec_tpab")
class SASRecTPABModel(SeqRecModel[SASRecTPABModelConfig, SASRecTPABModelOutput]):
    """SASRec model implementation.

    Here we implement a more advanced version of the SASRec model using: (1) RoPE
    for positional embeddings, (2) Pre-norm architecture with RMSNorm, (3) SwiGLU
    for feed-forward networks, and (4) 4x intermediate size in feed-forward networks.
    The overall architecture follows the original SASRec design and utilizes the
    implementations in Llama model.

    References:
        - Self-Attentive Sequential Recommendation. ICDM '18.
    """

    config_class = SASRecTPABModelConfig

    def __init__(self, config: SASRecTPABModelConfig) -> None:
        """Initializes SASRec model with the given configuration."""
        super().__init__(config)
        self.config: SASRecModelConfig

        assert (
            config.hidden_size % config.num_attention_heads == 0
        ), "hidden_size must be divisible by num_attention_heads."
        self.head_dim = config.hidden_size // config.num_attention_heads

        self.layers = nn.ModuleList(
            [
                LlamaDecoderLayer(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_attention_heads,
                    intermediate_size=config.hidden_size * 4,
                    attention_dropout=config.attention_dropout,
                    attention_bias=config.attention_bias,
                    ffn_bias=config.ffn_bias,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.rotary_emb = RotaryEmbedding(head_dim=self.head_dim)
        self.final_layer_norm = RMSNorm(config.hidden_size)

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
    ) -> SASRecTPABModelOutput:
        """Forward pass for SASRec model.

        Args:
            input_ids (Int[torch.Tensor, "B L"]): Input item ID sequences of shape (batch_size, seq_len).
            attention_mask (Int[torch.Tensor, "B L"]): Attention masks of shape (batch_size, seq_len).
            output_model_loss (bool): Whether to compute and return the model-specific loss. Default is False.
            output_hidden_states (bool): Whether to return hidden states from all layers. Default is False.
            output_attentions (bool): Whether to return attention weights from all layers. Default is False.
            **kwargs (Any): Additional keyword arguments for the model.

        Returns:
            SASRecModelOutput: Model outputs packaged as a `SASRecModelOutput` descendant.
        """
        timestamps: Optional[Int[torch.Tensor, "B L"]] = kwargs.pop("timestamps", None)
        if timestamps is None:  # pragma: no cover - defensive check
            raise ValueError("TPABModel.forward requires `timestamps` to be provided via kwargs.")

        hidden_states: Float[torch.Tensor, "B L d"]
        hidden_states = self.embed_tokens(input_ids)

        causal_mask: Float[torch.Tensor, "B 1 L L"]
        causal_mask = create_attention_mask(attention_mask, is_causal=True)

        position_embeddings: Tuple[Float[torch.Tensor, "B L head_dim"], Float[torch.Tensor, "B L head_dim"]]
        position_embeddings = self.rotary_emb(hidden_states)

        model_loss = None  # By default, SASRec does not compute model loss internally.
        all_hidden_states: List[Float[torch.Tensor, "B L d"]] = []
        all_attentions: List[Float[torch.Tensor, "B H L L"]] = []

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            hidden_states, attn_weights = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
            )

            if output_attentions:
                all_attentions.append(attn_weights)

        hidden_states = self.final_layer_norm(hidden_states)

        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        return SASRecTPABModelOutput(
            last_hidden_state=hidden_states,
            model_loss=model_loss if output_model_loss else None,
            hidden_states=tuple(all_hidden_states) if output_hidden_states else None,
            attentions=tuple(all_attentions) if output_attentions else None,
        )
