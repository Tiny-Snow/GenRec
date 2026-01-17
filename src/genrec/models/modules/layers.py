"""Standard layers."""

from __future__ import annotations

from typing import Optional, Tuple

from jaxtyping import Bool, Float, Int
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import MaskedHSTUAttention, MaskedSelfAttentionWithRoPE
from transformers.modeling_layers import GradientCheckpointingLayer
from .feedforward import SwiGLU
from .layernorm import RMSNorm
from .utils import create_attention_mask

__all__ = [
    "LlamaDecoderLayer",
    "SequentialTransductionUnit",
    "SpringLlamaDecoderLayer",
    "SpringSequentialTransductionUnit",
    "spring_attention_weight_spectral_norm",
    "spring_power_iteration",
]


class LlamaDecoderLayer(GradientCheckpointingLayer):
    """A standard Llama Transformer Decoder Layer, following `transformers.LlamaDecoderLayer`'s
    implementation."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        attention_dropout: float = 0.0,
        attention_bias: bool = False,
        ffn_bias: bool = False,
    ) -> None:
        """Initializes LlamaDecoderLayer module.

        Args:
            hidden_size (int): Dimensionality of the model's hidden representations.
            num_heads (int): Number of attention heads.
            intermediate_size (int): Dimensionality of the feed-forward network's intermediate layer.
            attention_dropout (float): Dropout rate for attention weights. Default is 0.0.
            attention_bias (bool): Whether to include bias terms in the attention projections. Default is False.
            ffn_bias (bool): Whether to include bias terms in the feed-forward network projections. Default is False.
        """
        super().__init__()

        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads."
        self.head_dim = hidden_size // num_heads
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.attention_dropout = attention_dropout
        self.attention_bias = attention_bias
        self.ffn_bias = ffn_bias

        self.self_attn = MaskedSelfAttentionWithRoPE(
            hidden_size=hidden_size,
            head_dim=self.head_dim,
            num_heads=num_heads,
            attention_dropout=attention_dropout,
            attention_bias=attention_bias,
        )
        self.mlp = SwiGLU(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            ffn_bias=ffn_bias,
        )
        self.input_layernorm = RMSNorm(hidden_size)
        self.post_attention_layernorm = RMSNorm(hidden_size)

    def forward(
        self,
        hidden_states: Float[torch.Tensor, "B L d"],
        attention_mask: Optional[Float[torch.Tensor, "B 1 L L"]] = None,
        position_embeddings: Optional[
            Tuple[Float[torch.Tensor, "B L head_dim"], Float[torch.Tensor, "B L head_dim"]]
        ] = None,
    ) -> Tuple[Float[torch.Tensor, "B L d"], Float[torch.Tensor, "B H L L"]]:
        """Forward pass for LlamaDecoderLayer.

        Args:
            hidden_states (Float[torch.Tensor, "B L d"]): Input tensor of shape (batch_size, seq_len, hidden_size).
            attention_mask (Optional[Float[torch.Tensor, "B 1 L L"]]): Optional attention mask added to the
                attention scores before softmax, where the masked positions are indicated by large negative values.
            position_embeddings (Optional[Tuple[Float[torch.Tensor, "B L head_dim"], Float[torch.Tensor, "B L head_dim"]]]):
                Optional tuple of cosine and sine embeddings for RoPE.

        Returns:
            Tuple[Float[torch.Tensor, "B L d"], Float[torch.Tensor, "B H L L"]]: A tuple containing the output
                tensor and the attention weights tensor.
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)

        return hidden_states, attn_weights


class SequentialTransductionUnit(GradientCheckpointingLayer):
    """Sequential Transduction Unit (STU) layer.

    Compared to standard STU layer, this module provides several options to generalize and
    enhance the performance of HSTU, including:

    - Option to switch the original learnable relative positional embeddings with Rotary Positional
        Embeddings (RoPE) for better extrapolation to longer sequences and improved performance.
    - Option to disable the original attention gating mechanism, allowing for a standard attention
        computation, which can be beneficial in certain scenarios where gating may not be stable.
    - Option to restore the FFN after attention, which was removed in the original STU design, to enhance
        the model's capacity.

    References:
        - Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for
            Generative Recommendations. ICML '24.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        attention_dropout: float = 0.0,
        attention_bias: bool = False,
        linear_dropout: float = 0.0,
        ffn_bias: bool = False,
        max_seq_len: int = 512,
        num_buckets: int = 128,
        enable_learnable_rel_posemb: bool = True,
        enable_attention_gating: bool = True,
        enable_ffn: bool = False,
    ) -> None:
        """Initializes SequentialTransductionUnit module.

        Args:
            hidden_size (int): Dimensionality of the model's hidden representations.
            num_heads (int): Number of attention heads.
            intermediate_size (int): Dimensionality of the feed-forward network's intermediate layer.
                Note that this is only used if `enable_ffn` is True.
            attention_dropout (float): Dropout rate for attention weights. Default is 0.0.
            attention_bias (bool): Whether to include bias terms in the attention projections. Default is False.
            linear_dropout (float): Dropout rate for the attention output before the final output projection, which
                is applied in `av_output` when attention gating is enabled. Note that this is only used if
                `enable_attention_gating` is True. Default is 0.0.
            ffn_bias (bool): Whether to include bias terms in the feed-forward network projections.
                Note that this is only used if `enable_ffn` is True. Default is False.
            max_seq_len (int): Maximum sequence length for relative positional embeddings. Default is 512.
            num_buckets (int): Number of buckets for relative positional embeddings. Default is 128.
            enable_learnable_rel_posemb (bool): Whether to use learnable relative positional embeddings.
                If False, RoPE will be used instead. Default is True.
            enable_attention_gating (bool): Whether to enable the attention gating mechanism. If False,
                standard attention computation is used. Default is True.
            enable_ffn (bool): Whether to include a feed-forward network after attention. Default is False.
        """
        super().__init__()

        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads."
        self.head_dim = hidden_size // num_heads
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.attention_dropout = attention_dropout
        self.attention_bias = attention_bias
        self.linear_dropout = linear_dropout
        self.ffn_bias = ffn_bias
        self.max_seq_len = max_seq_len
        self.num_buckets = num_buckets
        self.enable_learnable_rel_posemb = enable_learnable_rel_posemb
        self.enable_attention_gating = enable_attention_gating
        self.enable_ffn = enable_ffn

        self.self_attn = MaskedHSTUAttention(
            hidden_size=hidden_size,
            head_dim=self.head_dim,
            num_heads=num_heads,
            attention_dropout=attention_dropout,
            attention_bias=attention_bias,
            linear_dropout=linear_dropout,
            max_seq_len=max_seq_len,
            num_buckets=num_buckets,
            enable_learnable_rel_posemb=enable_learnable_rel_posemb,
            enable_attention_gating=enable_attention_gating,
        )
        self.input_layernorm = RMSNorm(hidden_size)

        self.mlp: Optional[SwiGLU] = None
        self.post_attention_layernorm: Optional[RMSNorm] = None
        if self.enable_ffn:
            self.mlp = SwiGLU(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                ffn_bias=ffn_bias,
            )
            self.post_attention_layernorm = RMSNorm(hidden_size)

    def forward(
        self,
        hidden_states: Float[torch.Tensor, "B L d"],
        attention_mask: Optional[Float[torch.Tensor, "B 1 L L"]] = None,
        position_embeddings: Optional[
            Tuple[Float[torch.Tensor, "B L head_dim"], Float[torch.Tensor, "B L head_dim"]]
        ] = None,
        timestamps: Optional[Int[torch.Tensor, "B L"]] = None,
    ) -> Tuple[Float[torch.Tensor, "B L d"], Float[torch.Tensor, "B H L L"]]:
        """Forward pass for SequentialTransductionUnit.

        Args:
            hidden_states (Float[torch.Tensor, "B L d"]): Input tensor of shape (batch_size, seq_len, hidden_size).
            attention_mask (Optional[Float[torch.Tensor, "B 1 L L"]]): Optional attention mask added to the attention
                scores before silu attention, where the masked positions are indicated by large negative values.
            position_embeddings (Optional[Tuple[Float[torch.Tensor, "B L head_dim"], Float[torch.Tensor, "B L head_dim"]]]):
                Optional tuple of cosine and sine embeddings for RoPE. Note that when `enable_learnable_rel_posemb` is True,
                this argument will be ignored.
            timestamps (Optional[Int[torch.Tensor, "B L"]]): Optional timestamps for each item in the sequence.

        Returns:
            Tuple[Float[torch.Tensor, "B L d"], Float[torch.Tensor, "B H L L"]]: A tuple containing the output
                tensor and the attention weights tensor.
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            timestamps=timestamps,
        )
        hidden_states = residual + hidden_states

        if self.enable_ffn and self.mlp is not None and self.post_attention_layernorm is not None:
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = residual + self.mlp(hidden_states)

        return hidden_states, attn_weights


def spring_power_iteration(
    module: nn.Module,
    W: Float[torch.Tensor, "m n"],
    name: str = "",
    spectral_norm_iters: int = 1,
    eps: float = 1e-12,
) -> Float[torch.Tensor, ""]:
    """Performs power iteration to estimate the largest singular value of a matrix.

    Args:
        module (nn.Module): The parent module to register the singular vectors as buffers.
        w (Float[torch.Tensor, "m n"]): The weight matrix, of shape (m, n).
        name (str): Name prefix for the registered buffers to store the singular vectors.
            If not empty, the left and right singular vectors will be registered as buffers
            under the names `{name}_u` and `{name}_v`. If empty, no buffers are registered.
            Default is "".
        spectral_norm_iters (int): Number of power iteration steps to perform. Default is 1.
        eps (float): Small value to avoid division by zero. Default is 1e-12.

    Returns:
        Float[torch.Tensor, ""]: Estimated largest singular value. The returned scalar tensor
            has gradients w.r.t. the input weight matrix `W`.

    .. warning::
        If `name` is provided, the left and right singular vectors are stored as buffers.
        However, during gradient checkpointing recomputations, the iterations should not be
        performed again to avoid inconsistent singular vectors or gradients. Therefore, this
        function should not be called in the checkpointed forward pass.

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
            vec = torch.randn(shape, device=device, dtype=dtype)
            vec = vec / (vec.norm() + eps)
        else:
            if not hasattr(module, buffer_name):
                vec = torch.randn(shape, device=device, dtype=dtype)
                vec = vec / (vec.norm() + eps)
                module.register_buffer(buffer_name, vec)
            vec = getattr(module, buffer_name)
        return vec.detach().clone()  # need clone, otherwise in-place update will fail in gradient backpropagation

    u: Float[torch.Tensor, "m"] = _get_singular_vector((m,), name + "_u" if name else "")
    v: Float[torch.Tensor, "n"] = _get_singular_vector((n,), name + "_v" if name else "")

    for _ in range(spectral_norm_iters):
        u = W @ v
        u = u / (u.norm() + eps)
        v = W.T @ u
        v = v / (v.norm() + eps)

    if name:
        # must be in-place copy to fit into DataParallel
        # see `torch.nn.utils.spectral_norm` for more details
        getattr(module, name + "_u").copy_(u.detach())
        getattr(module, name + "_v").copy_(v.detach())

    singular_value: Float[torch.Tensor, ""] = torch.norm(W @ v)  # equivalent to u^T W v

    return singular_value


def spring_attention_weight_spectral_norm(
    attn_weight: Float[torch.Tensor, "B L L"],
    tau: float,
    padding_mask: Optional[Int[torch.Tensor, "B L"]] = None,
) -> Float[torch.Tensor, ""]:
    """Estimates the spectral norm of the attention weight by its upper bound,
    i.e., 1-norm of the attention weight. Here we flatten the batch and sequence
    dimensions to one dimension, and estimate the 1-norm by log-sum-exp trick.
    The padding positions are masked out by the `attention_mask`.

    .. note::
        This function assumes that the attention weights are normalized along
        the last dimension (e.g., softmaxed). If the attention weights are not
        normalized, the exact upper bound should be:

        .. math::
            \\| A \\|_2 \\leq \\sqrt{\\| A \\|_1 \\| A \\|_{\\infty}}

        However, we do not observe significant performance difference when using
        the above upper bound in practice.

    Args:
        attn_weight (Float[torch.Tensor, "B L L"]): Attention weight tensor
            of shape (batch_size, seq_len, seq_len).
        tau (float): Temperature for the Spring regularization on attention module.
        padding_mask (Optional[Int[torch.Tensor, "B L"]]): Optional padding mask
            where 1 indicates valid tokens and 0 indicates padding tokens. This is
            used to generate the causal mask for attention weights. If None, no masking
            is applied. Default is None.

    Returns:
        Float[torch.Tensor, ""]: Estimated spectral norm of the attention weight.
    """
    if padding_mask is not None:
        # remask the attention weights to fix non-zero values at all-masked positions
        causal_mask: Bool[torch.Tensor, "B 1 L L"]
        causal_mask = create_attention_mask(padding_mask, is_causal=True, mask_value=1).bool()
        attn_weight = attn_weight.masked_fill(causal_mask.squeeze(1), 0.0)

        query_sums: Float[torch.Tensor, "B*L"] = attn_weight.sum(dim=-2).flatten()
        attention_mask_flat: Bool[torch.Tensor, "B*L"] = padding_mask.bool().flatten()
        masked_query_sums: Float[torch.Tensor, "M"] = query_sums[attention_mask_flat]
        norm_p1 = torch.logsumexp(masked_query_sums * tau, dim=0) / tau
    else:  # pragma: no cover - rarely used
        query_sums: Float[torch.Tensor, "B*L"] = attn_weight.sum(dim=-2).flatten()
        norm_p1 = torch.logsumexp(query_sums * tau, dim=0) / tau

    return norm_p1


# TODO: add my own paper about Spring regularization.
class SpringLlamaDecoderLayer(GradientCheckpointingLayer):
    """LlamaDecoderLayer with Spring regularization.

    See `LlamaDecoderLayer` for more details.

    References:
        - ...
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        attention_dropout: float = 0.0,
        attention_bias: bool = False,
        ffn_bias: bool = False,
        spring_attention_temperature: float = 1.0,
    ) -> None:
        """Initializes SpringLlamaDecoderLayer module.

        Args:
            hidden_size (int): Dimensionality of the model's hidden representations.
            num_heads (int): Number of attention heads.
            intermediate_size (int): Dimensionality of the feed-forward network's intermediate layer.
            attention_dropout (float): Dropout rate for attention weights. Default is 0.0.
            attention_bias (bool): Whether to include bias terms in the attention projections. Default is False.
            ffn_bias (bool): Whether to include bias terms in the feed-forward network projections. Default is False.
            apring_attention_temperature (float): Temperature for the Spring regularization on attention module.
                Default is 1.0.
        """
        super().__init__()

        self.spring_attention_temperature = spring_attention_temperature

        self._layer = LlamaDecoderLayer(
            hidden_size=hidden_size,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            attention_dropout=attention_dropout,
            attention_bias=attention_bias,
            ffn_bias=ffn_bias,
        )

    def forward(
        self,
        hidden_states: Float[torch.Tensor, "B L d"],
        padding_mask: Optional[Int[torch.Tensor, "B L"]] = None,
        attention_mask: Optional[Float[torch.Tensor, "B 1 L L"]] = None,
        position_embeddings: Optional[
            Tuple[Float[torch.Tensor, "B L head_dim"], Float[torch.Tensor, "B L head_dim"]]
        ] = None,
        output_model_loss: bool = False,
    ) -> Tuple[Float[torch.Tensor, "B L d"], Float[torch.Tensor, "B H L L"], Optional[Float[torch.Tensor, "H"]]]:
        """Forward pass for LlamaDecoderLayer with Spring regularization.

        Args:
            hidden_states (Float[torch.Tensor, "B L d"]): Input tensor of shape (batch_size, seq_len, hidden_size).
            padding_mask (Optional[Int[torch.Tensor, "B L"]]): Optional padding mask where 1 indicates valid tokens
                and 0 indicates padding tokens.
            attention_mask (Optional[Float[torch.Tensor, "B 1 L L"]]): Optional attention mask added to the
                attention scores before softmax, where the masked positions are indicated by large negative values.
            position_embeddings (Optional[Tuple[Float[torch.Tensor, "B L head_dim"], Float[torch.Tensor, "B L head_dim"]]]):
                Optional tuple of cosine and sine embeddings for RoPE.
            output_model_loss (bool): Whether to compute and return the model-specific loss (i.e., Spring loss).
                Default is False.

        Returns:
            Tuple[Float[torch.Tensor, "B L d"], Float[torch.Tensor, "B H L L"], Optional[Float[torch.Tensor, "H"]]]:
                A tuple containing the output tensor, the attention weights tensor, and optionally the Spring losses
                on attention weights (i.e., spectral norms for attention weights of each head).

        .. note::
            Note that the other components of Spring loss (e.g., item embedding Spring loss, FFN Spring loss, and
            attention V/O projection Spring loss) are computed outside this layer for the consistency with gradient
            checkpointing.
        """
        hidden_states, attn_weights = self._layer(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
        )

        # Spring regularizations on attention weights
        attn_weight_sn: Optional[Float[torch.Tensor, "H"]] = None
        if output_model_loss:
            attn_weight_sn = torch.zeros(self._layer.num_heads, device=hidden_states.device, dtype=hidden_states.dtype)
            for head in range(self._layer.num_heads):
                attn_weight_h: Float[torch.Tensor, "B L L"] = attn_weights[:, head, :, :]
                attn_weight_h_sn = spring_attention_weight_spectral_norm(
                    attn_weight_h,
                    tau=self.spring_attention_temperature,
                    padding_mask=padding_mask,
                )
                attn_weight_sn[head] = attn_weight_h_sn
        else:
            attn_weight_sn = None

        return hidden_states, attn_weights, attn_weight_sn


# TODO: add my own paper about Spring regularization.
class SpringSequentialTransductionUnit(GradientCheckpointingLayer):
    """Sequential Transduction Unit (STU) layer with Spring regularization.

    See `SequentialTransductionUnit` for more details.

    References:
        - Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for
            Generative Recommendations. ICML '24.
        - ...
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        attention_dropout: float = 0.0,
        attention_bias: bool = False,
        linear_dropout: float = 0.0,
        ffn_bias: bool = False,
        max_seq_len: int = 512,
        num_buckets: int = 128,
        enable_learnable_rel_posemb: bool = True,
        enable_attention_gating: bool = True,
        enable_ffn: bool = False,
        spring_attention_temperature: float = 1.0,
    ) -> None:
        """Initializes SpringSequentialTransductionUnit module.

        Args:
            hidden_size (int): Dimensionality of the model's hidden representations.
            num_heads (int): Number of attention heads.
            intermediate_size (int): Dimensionality of the feed-forward network's intermediate layer.
                Note that this is only used if `enable_ffn` is True.
            attention_dropout (float): Dropout rate for attention weights. Default is 0.0.
            attention_bias (bool): Whether to include bias terms in the attention projections. Default is False.
            linear_dropout (float): Dropout rate for the attention output before the final output projection, which
                is applied in `av_output` when attention gating is enabled. Note that this is only used if
                `enable_attention_gating` is True. Default is 0.0.
            ffn_bias (bool): Whether to include bias terms in the feed-forward network projections.
                Note that this is only used if `enable_ffn` is True. Default is False.
            max_seq_len (int): Maximum sequence length for relative positional embeddings. Default is 512.
            num_buckets (int): Number of buckets for relative positional embeddings. Default is 128.
            enable_learnable_rel_posemb (bool): Whether to use learnable relative positional embeddings.
                If False, RoPE will be used instead. Default is True.
            enable_attention_gating (bool): Whether to enable the attention gating mechanism. If False,
                standard attention computation is used. Default is True.
            enable_ffn (bool): Whether to include a feed-forward network after attention. Default is False.
            spring_attention_temperature (float): Temperature for the Spring regularization on attention module.
                Default is 1.0.
        """
        super().__init__()

        self.spring_attention_temperature = spring_attention_temperature

        self._layer = SequentialTransductionUnit(
            hidden_size=hidden_size,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            attention_dropout=attention_dropout,
            attention_bias=attention_bias,
            linear_dropout=linear_dropout,
            ffn_bias=ffn_bias,
            max_seq_len=max_seq_len,
            num_buckets=num_buckets,
            enable_learnable_rel_posemb=enable_learnable_rel_posemb,
            enable_attention_gating=enable_attention_gating,
            enable_ffn=enable_ffn,
        )

    def forward(
        self,
        hidden_states: Float[torch.Tensor, "B L d"],
        padding_mask: Optional[Int[torch.Tensor, "B L"]] = None,
        attention_mask: Optional[Float[torch.Tensor, "B 1 L L"]] = None,
        position_embeddings: Optional[
            Tuple[Float[torch.Tensor, "B L head_dim"], Float[torch.Tensor, "B L head_dim"]]
        ] = None,
        timestamps: Optional[Int[torch.Tensor, "B L"]] = None,
        output_model_loss: bool = False,
    ) -> Tuple[Float[torch.Tensor, "B L d"], Float[torch.Tensor, "B H L L"], Optional[Float[torch.Tensor, "H"]]]:
        """Forward pass for SequentialTransductionUnit with Spring regularization.

        Args:
            hidden_states (Float[torch.Tensor, "B L d"]): Input tensor of shape (batch_size, seq_len, hidden_size).
            padding_mask (Optional[Int[torch.Tensor, "B L"]]): Optional padding mask where 1 indicates valid tokens
                and 0 indicates padding tokens.
            attention_mask (Optional[Float[torch.Tensor, "B 1 L L"]]): Optional attention mask added to the attention
                scores before silu attention, where the masked positions are indicated by large negative values.
            position_embeddings (Optional[Tuple[Float[torch.Tensor, "B L head_dim"], Float[torch.Tensor, "B L head_dim"]]]):
                Optional tuple of cosine and sine embeddings for RoPE. Note that when `enable_learnable_rel_posemb` is True,
                this argument will be ignored.
            timestamps (Optional[Int[torch.Tensor, "B L"]]): Optional timestamps for each item in the sequence.
            output_model_loss (bool): Whether to compute and return the model-specific loss (i.e., Spring losses).
                Default is False.

        Returns:
            Tuple[Float[torch.Tensor, "B L d"], Float[torch.Tensor, "B H L L"], Optional[Float[torch.Tensor, "H"]]]:
                A tuple containing the output tensor, the attention weights tensor, and optionally the Spring losses
                on attention weights (i.e., spectral norms for attention weights of each head).

        .. note::
            Note that the other components of Spring loss (e.g., item embedding Spring loss, FFN Spring loss, and
            attention V/O projection Spring loss) are computed outside this layer for the consistency with gradient
            checkpointing.
        """
        hidden_states, attn_weights = self._layer(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            timestamps=timestamps,
        )

        # Spring regularizations on attention weights
        attn_weight_sn: Optional[Float[torch.Tensor, "H"]] = None
        if output_model_loss:
            attn_weight_sn = torch.zeros(self._layer.num_heads, device=hidden_states.device, dtype=hidden_states.dtype)
            for head in range(self._layer.num_heads):
                attn_weight_h: Float[torch.Tensor, "B L L"] = attn_weights[:, head, :, :]
                attn_weight_h_sn = spring_attention_weight_spectral_norm(
                    attn_weight_h,
                    tau=self.spring_attention_temperature,
                    padding_mask=padding_mask,
                )
                attn_weight_sn[head] = attn_weight_h_sn
        else:
            attn_weight_sn = None

        return hidden_states, attn_weights, attn_weight_sn
