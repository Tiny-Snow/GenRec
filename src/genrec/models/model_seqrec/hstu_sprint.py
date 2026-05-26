"""SeqRec Model: HSTU with SPRINT regularization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from jaxtyping import Bool, Float, Int
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules import (
    LearnableInputPositionalEmbedding,
    RMSNorm,
    RotaryEmbedding,
    SPRINTSequentialTransductionUnit,
    create_attention_mask,
    sprint_power_iteration,
)
from .base import SeqRecModel, SeqRecModelFactory, SeqRecModelConfigFactory, SeqRecOutputFactory
from .hstu import HSTUModelConfig, HSTUModelOutput

__all__ = [
    "HSTUSPRINTModel",
    "HSTUSPRINTModelConfig",
    "HSTUSPRINTModelOutput",
]


@SeqRecModelConfigFactory.register("hstu_sprint")
class HSTUSPRINTModelConfig(HSTUModelConfig):
    """Configuration class for HSTU model with SPRINT regularization, which
    extends `HSTUModelConfig`."""

    def __init__(
        self,
        sprint_attention_weight: float = 1.0,
        sprint_ffn_weight: float = 0.001,
        sprint_emb_weight: float = 0.1,
        sprint_attention_temperature: float = 4.0,
        spectral_norm_iters: int = 1,
        norm_embeddings: bool = False,
        **kwargs,
    ) -> None:
        """Initializes the configuration with model hyperparameters.

        Args:
            sprint_attention_weight (float): Weight for the SPRINT regularization on
                attention module. Default is 1.0.
            sprint_ffn_weight (float): Weight for the SPRINT regularization on feed-forward
                network module. Note that this is only effective when `enable_ffn` is True in
                the base `HSTUModelConfig`. Default is 0.001.
            sprint_emb_weight (float): Weight for the SPRINT regularization on item
                embedding matrix. Default is 0.1.
            sprint_attention_temperature (float): Temperature for the SPRINT regularization
                on attention module. Default is 4.0.
            spectral_norm_iters (int): Number of power iteration steps for spectral norm
                estimation. Default is 1.
            norm_embeddings (bool): Whether to L2-normalize the item embeddings before
                computing SPRINT regularization. Default is False.
            **kwargs (Any): Additional keyword arguments for the base `HSTUModelConfig`.
        """
        super().__init__(**kwargs)
        self.sprint_attention_weight = sprint_attention_weight
        self.sprint_ffn_weight = sprint_ffn_weight
        self.sprint_emb_weight = sprint_emb_weight
        self.sprint_attention_temperature = sprint_attention_temperature
        self.spectral_norm_iters = spectral_norm_iters
        self.norm_embeddings = norm_embeddings


@SeqRecOutputFactory.register("hstu_sprint")
@dataclass
class HSTUSPRINTModelOutput(HSTUModelOutput):
    """Output class for HSTU model with SPRINT regularization.

    The `HSTUModelOutput` extends the `HSTUModelOutput` without adding any
    additional attributes.
    """

    pass


@SeqRecModelFactory.register("hstu_sprint")
class HSTUSPRINTModel(SeqRecModel[HSTUSPRINTModelConfig, HSTUSPRINTModelOutput]):
    """HSTU model with SPRINT regularization.

    Here we reuse the `HSTUModel` network architecture and add SPRINT regularization
    as the `model_loss` in the `HSTUSPRINTModelOutput`.

    References:
    - Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for
        Generative Recommendations. ICML '24.
    - Mitigating Popularity Bias Amplification in Scaling Transformer-based Sequential
        Recommenders. KDD '26.
    """

    config_class = HSTUSPRINTModelConfig
    supports_gradient_checkpointing = True

    def __init__(self, config: HSTUSPRINTModelConfig) -> None:
        """Initializes HSTU model with the given configuration."""
        super().__init__(config)
        self.config: HSTUSPRINTModelConfig

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
                SPRINTSequentialTransductionUnit(
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
                    sprint_attention_temperature=config.sprint_attention_temperature,
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
    ) -> HSTUSPRINTModelOutput:
        """Forward pass for HSTU model with SPRINT regularization.

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
            HSTUSPRINTModelOutput: Model outputs packaged as a `HSTUSPRINTModelOutput` instance.
        """
        d = self.config.hidden_size
        H = self.config.num_attention_heads
        head_dim = self.head_dim
        iters = self.config.spectral_norm_iters

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

        if self.config.enable_input_pos_emb and self.input_pos_emb is not None:
            hidden_states = self.input_pos_emb(hidden_states)

        # SPRINT regularizations
        sprint_loss_emb = torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)
        sprint_loss_attn = torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)
        sprint_loss_ffn = torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)

        # SPRINT regularization on item embeddings
        if output_model_loss:
            item_embed_weight: Float[torch.Tensor, "I+1 d"] = self.item_embed_weight
            if self.config.norm_embeddings:
                item_embed_weight = F.normalize(item_embed_weight, p=2, dim=-1)
            item_emb_sn = sprint_power_iteration(
                self,
                item_embed_weight,
                name="item_embed_weight",
                spectral_norm_iters=self.config.spectral_norm_iters,
            )
            sprint_loss_emb = item_emb_sn.log1p()

        all_hidden_states: List[Float[torch.Tensor, "B L d"]] = []
        all_attentions: List[Float[torch.Tensor, "B H L L"]] = []

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            attn_weight_sn: Optional[Float[torch.Tensor, "H"]] = None
            hidden_states, attn_weights, attn_weight_sn = layer(
                hidden_states,
                padding_mask=attention_mask,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                timestamps=timestamps,
                output_model_loss=output_model_loss,
            )

            if output_attentions:
                all_attentions.append(attn_weights)

            # SPRINT regularizations on attention modules
            if output_model_loss and attn_weight_sn is not None:
                attn_Wo: Float[torch.Tensor, "d H*d_head"] = layer._layer.self_attn.o_proj.weight
                attn_Wo_sn = sprint_power_iteration(layer, attn_Wo, name=f"attn_wo", spectral_norm_iters=iters)

                attn_Wv: Float[torch.Tensor, "H d_head d"] = layer._layer.self_attn.v_proj.weight.view(H, head_dim, d)
                attn_Av_sn = torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)
                for head in range(H):
                    attn_Wv_h: Float[torch.Tensor, "d_head d"] = attn_Wv[head]
                    attn_Wv_h_sn = sprint_power_iteration(
                        layer, attn_Wv_h, name=f"attn_head_{head}_wv", spectral_norm_iters=iters
                    )
                    attn_weight_h_sn: Float[torch.Tensor, ""] = attn_weight_sn[head]
                    attn_Av_sn = attn_Av_sn + attn_weight_h_sn * attn_Wv_h_sn.pow(2)

                sprint_loss_attn = sprint_loss_attn + (attn_Av_sn.sqrt() * attn_Wo_sn).log1p()

            # SPRINT regularizations on feed-forward network modules
            if output_model_loss and self.config.enable_ffn and layer._layer.mlp is not None:
                ffn_W1 = layer._layer.mlp.up_proj.weight
                ffn_W2 = layer._layer.mlp.down_proj.weight
                ffn_W1_sn = sprint_power_iteration(layer, ffn_W1, name=f"ffn_w1", spectral_norm_iters=iters)
                ffn_W2_sn = sprint_power_iteration(layer, ffn_W2, name=f"ffn_w2", spectral_norm_iters=iters)
                sprint_loss_ffn = sprint_loss_ffn + (ffn_W1_sn * ffn_W2_sn).log1p()

        # normalize by number of layers
        sprint_loss_attn = sprint_loss_attn / self.config.num_hidden_layers
        sprint_loss_ffn = sprint_loss_ffn / self.config.num_hidden_layers

        # model loss: SPRINT regularization
        model_loss = (
            self.config.sprint_attention_weight * sprint_loss_attn
            + self.config.sprint_ffn_weight * sprint_loss_ffn
            + self.config.sprint_emb_weight * sprint_loss_emb
        )

        # NOTE: in the official HSTU code, there is no final layer norm,
        # as it apply L2 normalization after getting the final item representations.
        # Here we still keep an optional final layer norm for other loss functions' sake.
        if self.config.enable_final_layernorm and self.final_layernorm is not None:
            hidden_states = self.final_layernorm(hidden_states)

        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        return HSTUSPRINTModelOutput(
            last_hidden_state=hidden_states,
            model_loss=model_loss if output_model_loss else None,
            hidden_states=tuple(all_hidden_states) if output_hidden_states else None,
            attentions=tuple(all_attentions) if output_attentions else None,
        )
