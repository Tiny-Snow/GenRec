"""Standard layers."""

from __future__ import annotations

from typing import Optional, Tuple

from jaxtyping import Bool, Float, Int
import torch
import torch.nn as nn
from transformers.cache_utils import EncoderDecoderCache
from transformers.modeling_layers import GradientCheckpointingLayer

from .attention import MaskedHSTUAttention, MaskedSelfAttentionWithRoPE, T5Attention
from .feedforward import SwiGLU
from .layernorm import RMSNorm
from .utils import create_attention_mask

__all__ = [
    "LlamaDecoderLayer",
    "SPRINTLlamaDecoderLayer",
    "SequentialTransductionUnit",
    "SPRINTSequentialTransductionUnit",
    "T5Block",
    "sprint_attention_weight_spectral_norm",
    "sprint_power_iteration",
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


def sprint_power_iteration(
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


def sprint_attention_weight_spectral_norm(
    attn_weights: Float[torch.Tensor, "B H L L"],
    tau: float,
    padding_mask: Optional[Int[torch.Tensor, "B L"]] = None,
) -> Float[torch.Tensor, "H"]:
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
        attn_weights (Float[torch.Tensor, "B H L L"]): Attention weight tensor
            of shape (batch_size, num_heads, seq_len, seq_len).
        tau (float): Temperature for the SPRINT regularization on attention module.
        padding_mask (Optional[Int[torch.Tensor, "B L"]]): Optional padding mask
            where 1 indicates valid tokens and 0 indicates padding tokens. This is
            used to generate the causal mask for attention weights. If None, no masking
            is applied. Default is None.

    Returns:
        Float[torch.Tensor, "H"]: Estimated spectral norm of the attention weight for each head.
    """
    if padding_mask is not None:
        # remask the attention weights to fix non-zero values at all-masked positions
        causal_mask: Bool[torch.Tensor, "B 1 L L"]
        causal_mask = create_attention_mask(padding_mask, is_causal=True, mask_value=1).bool()
        attn_weights = attn_weights.masked_fill(causal_mask, 0.0)

        query_sums: Float[torch.Tensor, "H B*L"] = attn_weights.sum(dim=-2).permute(1, 0, 2).flatten(start_dim=1)
        attention_mask_flat: Bool[torch.Tensor, "B*L"] = padding_mask.bool().flatten()
        masked_query_sums: Float[torch.Tensor, "H M"] = query_sums[:, attention_mask_flat]
        norm_p1: Float[torch.Tensor, "H"] = torch.logsumexp(masked_query_sums * tau, dim=-1) / tau
    else:  # pragma: no cover - rarely used
        query_sums: Float[torch.Tensor, "H B*L"] = attn_weights.sum(dim=-2).permute(1, 0, 2).flatten(start_dim=1)
        norm_p1: Float[torch.Tensor, "H"] = torch.logsumexp(query_sums * tau, dim=-1) / tau

    return norm_p1


class SPRINTLlamaDecoderLayer(GradientCheckpointingLayer):
    """LlamaDecoderLayer with SPRINT regularization.

    See `LlamaDecoderLayer` for more details.

    References:
    - Mitigating Popularity Bias Amplification in Scaling Transformer-based Sequential
        Recommenders. KDD '26.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        attention_dropout: float = 0.0,
        attention_bias: bool = False,
        ffn_bias: bool = False,
        sprint_attention_temperature: float = 1.0,
    ) -> None:
        """Initializes SPRINTLlamaDecoderLayer module.

        Args:
            hidden_size (int): Dimensionality of the model's hidden representations.
            num_heads (int): Number of attention heads.
            intermediate_size (int): Dimensionality of the feed-forward network's intermediate layer.
            attention_dropout (float): Dropout rate for attention weights. Default is 0.0.
            attention_bias (bool): Whether to include bias terms in the attention projections. Default is False.
            ffn_bias (bool): Whether to include bias terms in the feed-forward network projections. Default is False.
            apring_attention_temperature (float): Temperature for the SPRINT regularization on attention module.
                Default is 1.0.
        """
        super().__init__()

        self.sprint_attention_temperature = sprint_attention_temperature

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
        """Forward pass for LlamaDecoderLayer with SPRINT regularization.

        Args:
            hidden_states (Float[torch.Tensor, "B L d"]): Input tensor of shape (batch_size, seq_len, hidden_size).
            padding_mask (Optional[Int[torch.Tensor, "B L"]]): Optional padding mask where 1 indicates valid tokens
                and 0 indicates padding tokens.
            attention_mask (Optional[Float[torch.Tensor, "B 1 L L"]]): Optional attention mask added to the
                attention scores before softmax, where the masked positions are indicated by large negative values.
            position_embeddings (Optional[Tuple[Float[torch.Tensor, "B L head_dim"], Float[torch.Tensor, "B L head_dim"]]]):
                Optional tuple of cosine and sine embeddings for RoPE.
            output_model_loss (bool): Whether to compute and return the model-specific loss (i.e., SPRINT loss).
                Default is False.

        Returns:
            Tuple[Float[torch.Tensor, "B L d"], Float[torch.Tensor, "B H L L"], Optional[Float[torch.Tensor, "H"]]]:
                A tuple containing the output tensor, the attention weights tensor, and optionally the SPRINT losses
                on attention weights (i.e., spectral norms for attention weights of each head).

        .. note::
            Note that the other components of SPRINT loss (e.g., item embedding SPRINT loss, FFN SPRINT loss, and
            attention V/O projection SPRINT loss) are computed outside this layer for the consistency with gradient
            checkpointing.
        """
        hidden_states, attn_weights = self._layer.forward(  # NOTE: to avoid meaningless recomputation in checkpointing
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
        )

        # SPRINT regularizations on attention weights
        attn_weight_sn: Optional[Float[torch.Tensor, "H"]] = None
        if output_model_loss:
            attn_weight_sn = sprint_attention_weight_spectral_norm(
                attn_weights,
                tau=self.sprint_attention_temperature,
                padding_mask=padding_mask,
            )
        else:
            attn_weight_sn = None

        return hidden_states, attn_weights, attn_weight_sn


class SPRINTSequentialTransductionUnit(GradientCheckpointingLayer):
    """Sequential Transduction Unit (STU) layer with SPRINT regularization.

    See `SequentialTransductionUnit` for more details.

    References:
    - Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for
        Generative Recommendations. ICML '24.
    - Mitigating Popularity Bias Amplification in Scaling Transformer-based Sequential
        Recommenders. KDD '26.
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
        sprint_attention_temperature: float = 1.0,
    ) -> None:
        """Initializes SPRINTSequentialTransductionUnit module.

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
            sprint_attention_temperature (float): Temperature for the SPRINT regularization on attention module.
                Default is 1.0.
        """
        super().__init__()

        self.sprint_attention_temperature = sprint_attention_temperature

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
        """Forward pass for SequentialTransductionUnit with SPRINT regularization.

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
            output_model_loss (bool): Whether to compute and return the model-specific loss (i.e., SPRINT losses).
                Default is False.

        Returns:
            Tuple[Float[torch.Tensor, "B L d"], Float[torch.Tensor, "B H L L"], Optional[Float[torch.Tensor, "H"]]]:
                A tuple containing the output tensor, the attention weights tensor, and optionally the SPRINT losses
                on attention weights (i.e., spectral norms for attention weights of each head).

        .. note::
            Note that the other components of SPRINT loss (e.g., item embedding SPRINT loss, FFN SPRINT loss, and
            attention V/O projection SPRINT loss) are computed outside this layer for the consistency with gradient
            checkpointing.
        """
        hidden_states, attn_weights = self._layer.forward(  # NOTE: to avoid meaningless recomputation in checkpointing
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            timestamps=timestamps,
        )

        # SPRINT regularizations on attention weights
        attn_weight_sn: Optional[Float[torch.Tensor, "H"]] = None
        if output_model_loss:
            attn_weight_sn = sprint_attention_weight_spectral_norm(
                attn_weights,
                tau=self.sprint_attention_temperature,
                padding_mask=padding_mask,
            )
        else:
            attn_weight_sn = None

        return hidden_states, attn_weights, attn_weight_sn


class T5Block(GradientCheckpointingLayer):
    """A standard T5 Transformer Block, following `transformers.T5Block`'s implementation.

    Compared to standard T5Block, our attention module provides several options to generalize
    and enhance the attention mechanism, including:
    - Option to switch the original learnable relative attention bias with Rotary Positional
    Embeddings (RoPE) for better extrapolation to longer sequences and improved performance.
    - We replace the original LayerNorm with RMSNorm for better training stability.
    - We replace the original feed-forward network with SwiGLU for improved model capacity.

    References:
    - Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. JMLR '20.
    """

    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_heads: int,
        intermediate_size: int,
        linear_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        attention_bias: bool = False,
        ffn_bias: bool = False,
        is_decoder: bool = False,
        has_relative_attention_bias: bool = False,
        relative_attention_num_buckets: int = 32,
        relative_attention_max_distance: int = 128,
        enable_rope: bool = False,
        layer_idx: Optional[int] = None,
    ) -> None:
        """Initializes T5Block module.

        Args:
            hidden_size (int): Dimensionality of the model's hidden representations.
            head_dim (int): Dimensionality of each attention head.
            num_heads (int): Number of attention heads.
            intermediate_size (int): Dimensionality of the feed-forward network's intermediate layer.
            linear_dropout (float): Dropout rate for the output of attention and feed-forward network. Default is 0.0.
            attention_dropout (float): Dropout rate for attention weights. Default is 0.0.
            attention_bias (bool): Whether to include bias terms in the attention projections. Default is False.
            ffn_bias (bool): Whether to include bias terms in the feed-forward network projections. Default is False.
            is_decoder (bool): Whether this attention module is used in the decoder. This is used to determine
                the directionality of relative positional embeddings. Default is False.
            has_relative_attention_bias (bool): Whether to compute learnable relative positional bias. If False, this
                module will not initialize a `T5RelativePositionBias` instance. Typically, T5 set `has_relative_attention_bias`
                to True only for the first block, while the rest blocks reuse the same relative positional bias. Note
                that when `enable_rope` is True, this argument will be ignored. Default is False.
            relative_attention_num_buckets (int): Number of buckets for relative positional embeddings. Default is 32.
            relative_attention_max_distance (int): Maximum distance for relative positional embeddings. Default is 128.
            enable_rope (bool): Whether to use RoPE instead of learnable relative positional bias. If False, the original
                learnable relative positional bias in T5 will be used. Default is False.
            layer_idx (Optional[int]): Optional layer index of this attention module in the model. This should be set
                when caching past key/values in the decoder for autoregressive generation. Default is None.
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.attention_dropout = attention_dropout
        self.attention_bias = attention_bias
        self.ffn_bias = ffn_bias
        self.is_decoder = is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.enable_rope = enable_rope
        self.layer_idx = layer_idx

        self.self_attn = T5Attention(
            hidden_size=hidden_size,
            head_dim=head_dim,
            num_heads=num_heads,
            attention_dropout=attention_dropout,
            attention_bias=attention_bias,
            is_decoder=is_decoder,
            has_relative_attention_bias=has_relative_attention_bias,
            relative_attention_num_buckets=relative_attention_num_buckets,
            relative_attention_max_distance=relative_attention_max_distance,
            enable_rope=enable_rope,
            layer_idx=layer_idx,
        )
        self.self_attn_layernorm = RMSNorm(hidden_size)
        self.self_attn_dropout = nn.Dropout(linear_dropout)

        self.cross_attn: Optional[T5Attention] = None
        self.cross_attn_layernorm: Optional[RMSNorm] = None
        self.cross_attn_dropout: Optional[nn.Dropout] = None
        if is_decoder:
            # cross attention disables relative positional embeddings or RoPE
            self.cross_attn = T5Attention(
                hidden_size=hidden_size,
                head_dim=head_dim,
                num_heads=num_heads,
                attention_dropout=attention_dropout,
                attention_bias=attention_bias,
                is_decoder=is_decoder,
                has_relative_attention_bias=False,
                enable_rope=False,
                layer_idx=layer_idx,
            )
            self.cross_attn_layernorm = RMSNorm(hidden_size)
            self.cross_attn_dropout = nn.Dropout(linear_dropout)

        self.mlp = SwiGLU(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            ffn_bias=ffn_bias,
            dropout=linear_dropout,
        )
        self.post_attention_layernorm = RMSNorm(hidden_size)

    def forward(
        self,
        hidden_states: Float[torch.Tensor, "B L d"],
        attention_mask: Optional[Float[torch.Tensor, "B 1 #L L_k"]] = None,
        position_bias: Optional[Float[torch.Tensor, "#B H L L"]] = None,
        position_embeddings: Optional[
            Tuple[Float[torch.Tensor, "B L head_dim"], Float[torch.Tensor, "B L head_dim"]]
        ] = None,
        encoder_hidden_states: Optional[Float[torch.Tensor, "B L_enc d"]] = None,
        encoder_attention_mask: Optional[Float[torch.Tensor, "B 1 1 L_enc"]] = None,
        encoder_decoder_position_bias: Optional[Float[torch.Tensor, "#B H L L_enc"]] = None,
        past_key_values: Optional[EncoderDecoderCache] = None,
        cache_position: Optional[Int[torch.Tensor, "L"]] = None,
        output_attentions: bool = False,
    ) -> Tuple[
        Float[torch.Tensor, "B L d"],
        Tuple[
            Optional[Float[torch.Tensor, "#B H L L"]],
            Optional[Float[torch.Tensor, "B H L L"]],
        ],
        Tuple[
            Optional[Float[torch.Tensor, "#B H L L_enc"]],
            Optional[Float[torch.Tensor, "B H L L_enc"]],
        ],
    ]:
        """Forward pass for T5Block.

        Args:
            hidden_states (Float[torch.Tensor, "B L d"]): Input tensor of shape (batch_size, seq_len, hidden_size).
                Note that for cross-attention in the decoder, the encoder output should be provided in the argument
                `encoder_hidden_states`.
            attention_mask (Optional[Float[torch.Tensor, "B 1 #L L_k"]]): Optional attention mask added to the self-attention
                scores before softmax, with shape either (batch_size, 1, seq_len, key_len) for causal self-attention in
                decoder, or (batch_size, 1, 1, key_len) for self-attention in encoder. Specifically, if the dimension
                `L_k` is longer than the actual key length `L` (which can happen during autoregressive generation in the
                decoder), the mask will be sliced accordingly. For cross-attention in the decoder, please use
                `encoder_attention_mask`. Default is None.
            position_bias (Optional[Float[torch.Tensor, "#B H L L"]]): Optional precomputed position bias to be added
                to the attention scores. If None, and if `has_relative_attention_bias` is True and `enable_rope` is False,
                the position bias will be computed based on the relative positions of the queries and keys from the T5
                learnable relative positional embeddings. Default is None.
            position_embeddings (Optional[Tuple[Float[torch.Tensor, "B L head_dim"], Float[torch.Tensor, "B L head_dim"]]]):
                Optional tuple of cosine and sine embeddings for RoPE. Note that when `enable_rope` is False, this argument
                will be ignored. Default is None.
            encoder_hidden_states (Optional[Float[torch.Tensor, "B L_enc d"]]): Optional encoder hidden states for
                cross-attention in the decoder, of shape (batch_size, encoder_len, hidden_size). If provided, cross-attention
                will be performed using these encoder hidden states as keys and values. Default is None.
            encoder_attention_mask (Optional[Float[torch.Tensor, "B 1 1 L_enc"]]): Optional attention mask for
                cross-attention in the decoder, of shape (batch_size, 1, 1, encoder_len). Default is None.
            encoder_decoder_position_bias (Optional[Float[torch.Tensor, "#B H L L_enc"]]): Optional precomputed
                position bias to be added to the cross-attention scores. If None, no positional bias will be used in
                cross-attention. In usual T5 implementations, cross-attention does not use relative positional embeddings
                or RoPE, and this argument is typically set to None. This argument can be provided manually if needed.
                Default is None.
            past_key_values (Optional[EncoderDecoderCache]): Optional cache for previously computed key and value states,
                used in the decoder for faster autoregressive generation. Default is None.
            cache_position (Optional[Int[torch.Tensor, "L"]]): Optional position IDs used to compute relative positions
                when caching past key/values in the decoder. If provided, it should contain the absolute positions of the
                current query tokens. Default is None.
            output_attentions (bool): Whether to return the attention weights. Default is False.

        Returns:
            Tuple[
                Float[torch.Tensor, "B L d"],
                Tuple[
                    Optional[Float[torch.Tensor, "#B H L L"]],
                    Optional[Float[torch.Tensor, "B H L L"]],
                ],
                Tuple[
                    Optional[Float[torch.Tensor, "#B H L L_enc"]],
                    Optional[Float[torch.Tensor, "B H L L_enc"]],
                ],
            ]: A tuple containing:
                - output tensor of shape (batch_size, seq_len, hidden_size).
                - a tuple of optional self-attention position bias and attention weights.
                - a tuple of optional cross-attention position bias and attention weights.
        """
        # Self-Attention
        residual = hidden_states
        hidden_states = self.self_attn_layernorm(hidden_states)
        self_attn_outputs: Tuple = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            position_embeddings=position_embeddings,
            past_key_values=past_key_values,
            cache_position=cache_position,
            output_attentions=output_attentions,
        )
        hidden_states = residual + self.self_attn_dropout(self_attn_outputs[0])
        self_attn_outputs = self_attn_outputs[1:]  # remove the output hidden states

        # Cross-Attention
        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        cross_attn_outputs = (None, None)
        if do_cross_attention:
            assert self.cross_attn is not None
            assert self.cross_attn_layernorm is not None
            assert self.cross_attn_dropout is not None

            residual = hidden_states
            hidden_states = self.cross_attn_layernorm(hidden_states)
            cross_attn_outputs = self.cross_attn(
                hidden_states,
                attention_mask=encoder_attention_mask,
                key_value_states=encoder_hidden_states,
                position_bias=encoder_decoder_position_bias,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
            )
            hidden_states = residual + self.cross_attn_dropout(cross_attn_outputs[0])
            cross_attn_outputs = cross_attn_outputs[1:]  # remove the output hidden states

        # Feed-Forward Network
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)

        return hidden_states, self_attn_outputs, cross_attn_outputs
