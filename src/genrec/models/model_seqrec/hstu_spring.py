"""SeqRec Model: HSTU with Spring regularization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, cast

from jaxtyping import Bool, Float, Int
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..modules import (
    LearnableInputPositionalEmbedding,
    RMSNorm,
    RotaryEmbedding,
    SequentialTransductionUnit,
    create_attention_mask,
)
from .base import SeqRecModel, SeqRecModelFactory, SeqRecModelConfigFactory, SeqRecOutputFactory
from .hstu import HSTUModelConfig, HSTUModelOutput

__all__ = [
    "HSTUSpringModel",
    "HSTUSpringModelConfig",
    "HSTUSpringModelOutput",
]


@SeqRecModelConfigFactory.register("hstu_spring")
class HSTUSpringModelConfig(HSTUModelConfig):
    """Configuration class for HSTU model with Spring regularization, which
    extends `HSTUModelConfig`."""

    def __init__(
        self,
        spring_attention_weight: float = 1.0,
        spring_ffn_weight: float = 0.001,
        spring_emb_weight: float = 0.1,
        spring_attention_temperature: float = 4.0,
        spectral_norm_iters: int = 1,
        norm_embeddings: bool = False,
        **kwargs,
    ) -> None:
        """Initializes the configuration with model hyperparameters.

        Args:
            spring_attention_weight (float): Weight for the Spring regularization on
                attention module. Default is 1.0.
            spring_ffn_weight (float): Weight for the Spring regularization on feed-forward
                network module. Note that this is only effective when `enable_ffn` is True in
                the base `HSTUModelConfig`. Default is 0.001.
            spring_emb_weight (float): Weight for the Spring regularization on item
                embedding matrix. Default is 0.1.
            spring_attention_temperature (float): Temperature for the Spring regularization
                on attention module. Default is 4.0.
            spectral_norm_iters (int): Number of power iteration steps for spectral norm
                estimation. Default is 1.
            norm_embeddings (bool): Whether to L2-normalize the item embeddings before
                computing Spring regularization. Default is False.
            **kwargs (Any): Additional keyword arguments for the base `HSTUModelConfig`.
        """
        super().__init__(**kwargs)
        self.spring_attention_weight = spring_attention_weight
        self.spring_ffn_weight = spring_ffn_weight
        self.spring_emb_weight = spring_emb_weight
        self.spring_attention_temperature = spring_attention_temperature
        self.spectral_norm_iters = spectral_norm_iters
        self.norm_embeddings = norm_embeddings


@SeqRecOutputFactory.register("hstu_spring")
@dataclass
class HSTUSpringModelOutput(HSTUModelOutput):
    """Output class for HSTU model with Spring regularization.

    The `HSTUModelOutput` extends the `HSTUModelOutput` without adding any
    additional attributes.
    """

    pass


# TODO: add my own paper about Spring regularization.
@SeqRecModelFactory.register("hstu_spring")
class HSTUSpringModel(SeqRecModel[HSTUSpringModelConfig, HSTUSpringModelOutput]):
    """HSTU model with Spring regularization.

    Here we reuse the `HSTUModel` network architecture and add Spring regularization
    as the `model_loss` in the `HSTUSpringModelOutput`.

    References:
        - Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for
            Generative Recommendations. ICML '24.
        - ...
    """

    config_class = HSTUSpringModelConfig

    def __init__(self, config: HSTUSpringModelConfig) -> None:
        """Initializes HSTU model with the given configuration."""
        super().__init__(config)
        self.config: HSTUSpringModelConfig

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
    ) -> HSTUSpringModelOutput:
        """Forward pass for HSTU model with Spring regularization.

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
            HSTUSpringModelOutput: Model outputs packaged as a `HSTUSpringModelOutput` instance.
        """
        d, H = self.config.hidden_size, self.config.num_attention_heads

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

        # Spring regularizations
        spring_loss_emb = torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)
        spring_loss_attn = torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)
        spring_loss_ffn = torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)

        # Spring regularization on item embeddings
        if output_model_loss:
            item_embed_weight: Float[torch.Tensor, "I+1 d"] = self.item_embed_weight
            if self.config.norm_embeddings:
                item_embed_weight = F.normalize(item_embed_weight, p=2, dim=-1)
            item_emb_sn = self._power_iteration(item_embed_weight, name="item_embed_weight")
            spring_loss_emb = item_emb_sn.log1p()

        all_hidden_states: List[Float[torch.Tensor, "B L d"]] = []
        all_attentions: List[Float[torch.Tensor, "B H L L"]] = []

        for layer_idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            def layer_forward(hidden_states: Float[torch.Tensor, "B L d"]):
                return layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_embeddings=position_embeddings,
                    timestamps=timestamps,
                )

            if self.gradient_checkpointing and self.training:
                hidden_states, attn_weights = checkpoint(  # pyright: ignore[reportGeneralTypeIssues]
                    layer_forward,
                    hidden_states,
                    use_reentrant=False,
                )
            else:
                hidden_states, attn_weights = layer_forward(hidden_states)

            if output_attentions:
                all_attentions.append(attn_weights)

            if output_model_loss:
                # Spring regularization on o_proj
                attn_Wo: Float[torch.Tensor, "d H*d_head"] = layer.self_attn.o_proj.weight
                attn_Wo_sn = self._power_iteration(attn_Wo, name=f"attn_{layer_idx}_wo")

                # Spring regularization on v_proj and attn weights
                attn_Wv: Float[torch.Tensor, "H d_head d"] = layer.self_attn.v_proj.weight.view(H, self.head_dim, d)
                attn_Av_sn = torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)
                for head in range(H):
                    attn_Wv_h: Float[torch.Tensor, "d_head d"] = attn_Wv[head]
                    attn_Wv_h_sn = self._power_iteration(attn_Wv_h, name=f"attn_{layer_idx}_head_{head}_wv")
                    attn_weight_h: Float[torch.Tensor, "B L L"] = attn_weights[:, head, :, :]
                    attn_weight_h_sn = self._attention_weight_spectral_norm(attn_weight_h, attention_mask)
                    attn_Av_sn = attn_Av_sn + attn_weight_h_sn * attn_Wv_h_sn.pow(2)

                # Spring regularization on attention module
                spring_loss_attn = spring_loss_attn + (attn_Av_sn.sqrt() * attn_Wo_sn).log1p()

                # Spring regularization on FFN modules
                if self.config.enable_ffn:
                    ffn_W1 = layer.mlp.up_proj.weight
                    ffn_W2 = layer.mlp.down_proj.weight
                    ffn_W1_sn = self._power_iteration(ffn_W1, name=f"ffn_{layer_idx}_w1")
                    ffn_W2_sn = self._power_iteration(ffn_W2, name=f"ffn_{layer_idx}_w2")
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

        # NOTE: in the official HSTU code, there is no final layer norm,
        # as it apply L2 normalization after getting the final item representations.
        # Here we still keep an optional final layer norm for other loss functions' sake.
        if self.config.enable_final_layernorm and self.final_layernorm is not None:
            hidden_states = self.final_layernorm(hidden_states)

        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        return HSTUSpringModelOutput(
            last_hidden_state=hidden_states,
            model_loss=model_loss if output_model_loss else None,
            hidden_states=tuple(all_hidden_states) if output_hidden_states else None,
            attentions=tuple(all_attentions) if output_attentions else None,
        )

    def _power_iteration(
        self,
        W: Float[torch.Tensor, "m n"],
        name: str = "",
        eps: float = 1e-12,
    ) -> Float[torch.Tensor, ""]:
        """Performs power iteration to estimate the largest singular value of a matrix.

        Args:
            w (Float[torch.Tensor, "m n"]): The weight matrix, of shape (m, n).
            name (str): Name prefix for the registered buffers to store the singular vectors.
            If not empty, the left and right singular vectors will be registered as buffers
            under the names `{name}_u` and `{name}_v`. If empty, no buffers
            are registered. Default is "".
            eps (float): Small value to avoid division by zero. Default is 1e-12.

        Returns:
            Float[torch.Tensor, ""]: Estimated largest singular value. The returned scalar tensor
                has gradients w.r.t. the input weight matrix `W`.

        References:
            - Spectral Norm Regularization for Improving the Generalizability of Deep Learning. arXiv '17.
            - Spectral Normalization for Generative Adversarial Networks. ICLR '18.
        """
        assert W.dim() == 2, "Input weight matrix must be 2-dimensional."
        m, n = W.shape
        device, dtype = W.device, W.dtype

        def _get_singular_vector(shape: Tuple[int, ...], buffer_name: str) -> Float[torch.Tensor, "..."]:
            """Fetches or initializes a singular vector. If `buffer_name` is empty,
            a non-registered tensor is returned.
            """
            if not buffer_name:
                vec = torch.randn(shape, device=device, dtype=dtype)  # no need to normalize here
            else:
                if not hasattr(self, buffer_name):
                    vec = torch.randn(shape, device=device, dtype=dtype)
                    vec = vec / (vec.norm() + eps)
                    self.register_buffer(buffer_name, vec)
                vec = getattr(self, buffer_name)
            return vec.detach().clone()  # need clone, otherwise in-place update will fail in gradient backpropagation

        u: Float[torch.Tensor, "m"] = _get_singular_vector((m,), name + "_u" if name else "")
        v: Float[torch.Tensor, "n"] = _get_singular_vector((n,), name + "_v" if name else "")

        for _ in range(self.config.spectral_norm_iters):
            u = W @ v
            u = u / (u.norm() + eps)
            v = W.T @ u
            v = v / (v.norm() + eps)

        if name:
            # must be in-place copy to fit into DataParallel
            # see `torch.nn.utils.spectral_norm` for more details
            getattr(self, name + "_u").copy_(u.detach())
            getattr(self, name + "_v").copy_(v.detach())

        singular_value: Float[torch.Tensor, ""] = torch.norm(W @ v)  # equivalent to u^T W v

        return singular_value

    def _attention_weight_spectral_norm(
        self,
        attn_weight: Float[torch.Tensor, "B L L"],
        attention_mask: Int[torch.Tensor, "B L"],
    ) -> Float[torch.Tensor, ""]:
        """Estimates the spectral norm of the attention weight by its upper bound,
        i.e., 1-norm of the attention weight. Here we flatten the batch and sequence
        dimensions to one dimension, and estimate the 1-norm by log-sum-exp trick.
        The padding positions are masked out by the `attention_mask`.

        .. note::
            The SiLU-based attention in HSTU makes the attention weights non-normalized
            in each row, so the infinity-norm is not guaranteed to be 1. However, we
            did not observe significant performance difference when adding the infinity-norm
            term in our experiments.

        Args:
            attn_weight (Float[torch.Tensor, "B L L"]): Attention weight tensor
                of shape (batch_size, seq_len, seq_len).
            attention_mask (Int[torch.Tensor, "B L"]): Attention mask tensor
                of shape (batch_size, seq_len).

        Returns:
            Float[torch.Tensor, ""]: Estimated spectral norm of the attention weight.
        """
        # remask the attention weights to fix non-zero values at all-masked positions
        causal_mask: Bool[torch.Tensor, "B 1 L L"]
        causal_mask = create_attention_mask(attention_mask, is_causal=True, mask_value=1).bool()
        attn_weight = attn_weight.masked_fill(causal_mask.squeeze(1), 0.0)

        query_sums: Float[torch.Tensor, "B*L"] = attn_weight.sum(dim=-2).flatten()
        attention_mask_flat: Bool[torch.Tensor, "B*L"] = attention_mask.bool().flatten()
        masked_query_sums: Float[torch.Tensor, "M"] = query_sums[attention_mask_flat]

        tau = self.config.spring_attention_temperature
        norm_p1 = torch.logsumexp(masked_query_sums * tau, dim=0) / tau

        return norm_p1
