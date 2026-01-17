"""SeqRec Model: SASRec with Spring regularization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from jaxtyping import Float, Int
import torch
import torch.nn as nn

from ..modules import SpringLlamaDecoderLayer, RMSNorm, RotaryEmbedding, create_attention_mask, spring_power_iteration
from .base import SeqRecModel, SeqRecModelConfigFactory, SeqRecModelFactory, SeqRecOutputFactory
from .sasrec import SASRecModelConfig, SASRecModelOutput

__all__ = [
    "SASRecSpringModel",
    "SASRecSpringModelConfig",
    "SASRecSpringModelOutput",
]


@SeqRecModelConfigFactory.register("sasrec_spring")
class SASRecSpringModelConfig(SASRecModelConfig):
    """Configuration class for SASRec model with Spring regularization, which
    extends `SASRecModelConfig`."""

    def __init__(
        self,
        spring_attention_weight: float = 1.0,
        spring_ffn_weight: float = 0.001,
        spring_emb_weight: float = 0.1,
        spring_attention_temperature: float = 4.0,
        spectral_norm_iters: int = 1,
        **kwargs,
    ) -> None:
        """Initializes the configuration with model hyperparameters.

        Args:
            spring_attention_weight (float): Weight for the Spring regularization on
                attention module. Default is 1.0.
            spring_ffn_weight (float): Weight for the Spring regularization on feed-forward
                network module. Default is 0.001.
            spring_emb_weight (float): Weight for the Spring regularization on item
                embedding matrix. Default is 0.1.
            spring_attention_temperature (float): Temperature for the Spring regularization
                on attention module. Default is 4.0.
            spectral_norm_iters (int): Number of power iteration steps for spectral norm
                estimation. Default is 1.
            **kwargs (Any): Additional keyword arguments for the base `SASRecModelConfig`.
        """
        super().__init__(**kwargs)
        self.spring_attention_weight = spring_attention_weight
        self.spring_ffn_weight = spring_ffn_weight
        self.spring_emb_weight = spring_emb_weight
        self.spring_attention_temperature = spring_attention_temperature
        self.spectral_norm_iters = spectral_norm_iters


@SeqRecOutputFactory.register("sasrec_spring")
@dataclass
class SASRecSpringModelOutput(SASRecModelOutput):
    """Output class for SASRec model with Spring regularization.

    The `SASRecSpringModelOutput` extends the `SASRecModelOutput`
    without adding any additional attributes.
    """

    pass


# TODO: add my own paper about Spring regularization.
@SeqRecModelFactory.register("sasrec_spring")
class SASRecSpringModel(SeqRecModel[SASRecSpringModelConfig, SASRecSpringModelOutput]):
    """SASRec model with Spring regularization.

    Here we reuse the `SASRecModel` network architecture and add Spring regularization
    as the `model_loss` in the `SASRecSpringModelOutput`.

    Reference:
        - Self-Attentive Sequential Recommendation. ICDM '18.
        - ...
    """

    config_class = SASRecSpringModelConfig

    def __init__(self, config: SASRecSpringModelConfig) -> None:
        """Initializes SASRec model with the given configuration."""
        super().__init__(config)
        self.config: SASRecSpringModelConfig

        assert (
            config.hidden_size % config.num_attention_heads == 0
        ), "hidden_size must be divisible by num_attention_heads."
        self.head_dim = config.hidden_size // config.num_attention_heads

        self.layers = nn.ModuleList(
            [
                SpringLlamaDecoderLayer(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_attention_heads,
                    intermediate_size=config.hidden_size * 4,
                    attention_dropout=config.attention_dropout,
                    attention_bias=config.attention_bias,
                    ffn_bias=config.ffn_bias,
                    spring_attention_temperature=config.spring_attention_temperature,
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
    ) -> SASRecSpringModelOutput:
        """Forward pass for SASRec model with Spring regularization.

        Args:
            input_ids (Int[torch.Tensor, "B L"]): Input item ID sequences of shape (batch_size, seq_len).
            attention_mask (Int[torch.Tensor, "B L"]): Attention masks of shape (batch_size, seq_len).
            output_model_loss (bool): Whether to compute and return the model-specific loss. Default is False.
            output_hidden_states (bool): Whether to return hidden states from all layers. Default is False.
            output_attentions (bool): Whether to return attention weights from all layers. Default is False.
            **kwargs (Any): Additional keyword arguments for the model.

        Returns:
            SASRecSpringModelOutput: Model outputs packaged as a `SASRecSpringModelOutput` descendant.
        """
        d = self.config.hidden_size
        H = self.config.num_attention_heads
        head_dim = self.head_dim
        iters = self.config.spectral_norm_iters

        hidden_states: Float[torch.Tensor, "B L d"]
        hidden_states = self.embed_tokens(input_ids)

        causal_mask: Float[torch.Tensor, "B 1 L L"]
        causal_mask = create_attention_mask(attention_mask, is_causal=True)

        position_embeddings: Tuple[Float[torch.Tensor, "B L head_dim"], Float[torch.Tensor, "B L head_dim"]]
        position_embeddings = self.rotary_emb(hidden_states)

        # Spring regularizations
        spring_loss_emb = torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)
        spring_loss_attn = torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)
        spring_loss_ffn = torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)

        # Spring regularization on item embeddings
        if output_model_loss:
            item_emb_sn = spring_power_iteration(
                self,
                self.item_embed_weight,
                name="item_embed_weight",
                spectral_norm_iters=self.config.spectral_norm_iters,
            )
            spring_loss_emb = item_emb_sn.log1p()

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
                output_model_loss=output_model_loss,
            )

            if output_attentions:
                all_attentions.append(attn_weights)

            # Spring regularizations on attention modules
            if output_model_loss and attn_weight_sn is not None:
                attn_Wo: Float[torch.Tensor, "d H*d_head"] = layer._layer.self_attn.o_proj.weight
                attn_Wo_sn = spring_power_iteration(layer, attn_Wo, name=f"attn_wo", spectral_norm_iters=iters)

                attn_Wv: Float[torch.Tensor, "H d_head d"] = layer._layer.self_attn.v_proj.weight.view(H, head_dim, d)
                attn_Av_sn = torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)
                for head in range(H):
                    attn_Wv_h: Float[torch.Tensor, "d_head d"] = attn_Wv[head]
                    attn_Wv_h_sn = spring_power_iteration(
                        layer, attn_Wv_h, name=f"attn_head_{head}_wv", spectral_norm_iters=iters
                    )
                    attn_weight_h_sn: Float[torch.Tensor, ""] = attn_weight_sn[head]
                    attn_Av_sn = attn_Av_sn + attn_weight_h_sn * attn_Wv_h_sn.pow(2)

                spring_loss_attn = spring_loss_attn + (attn_Av_sn.sqrt() * attn_Wo_sn).log1p()

            # Spring regularizations on feed-forward network modules
            if output_model_loss:
                ffn_W1 = layer._layer.mlp.up_proj.weight
                ffn_W2 = layer._layer.mlp.down_proj.weight
                ffn_W1_sn = spring_power_iteration(layer, ffn_W1, name=f"ffn_w1", spectral_norm_iters=iters)
                ffn_W2_sn = spring_power_iteration(layer, ffn_W2, name=f"ffn_w2", spectral_norm_iters=iters)
                spring_loss_ffn = spring_loss_ffn + (ffn_W1_sn * ffn_W2_sn).log1p()

        # normalize by number of layers
        spring_loss_attn = spring_loss_attn / self.config.num_hidden_layers
        spring_loss_ffn = spring_loss_ffn / self.config.num_hidden_layers

        # model loss: Spring regularization
        model_loss = (
            self.config.spring_attention_weight * spring_loss_attn
            + self.config.spring_ffn_weight * spring_loss_ffn
            + self.config.spring_emb_weight * spring_loss_emb
        )

        hidden_states = self.final_layer_norm(hidden_states)

        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        return SASRecSpringModelOutput(
            last_hidden_state=hidden_states,
            model_loss=model_loss if output_model_loss else None,
            hidden_states=tuple(all_hidden_states) if output_hidden_states else None,
            attentions=tuple(all_attentions) if output_attentions else None,
        )
