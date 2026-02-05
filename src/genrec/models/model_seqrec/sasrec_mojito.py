"""SeqRec Model: SASRec."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from jaxtyping import Float, Int
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules import LlamaDecoderLayer, RMSNorm, RotaryEmbedding, create_attention_mask
from .base import (
    SeqRecModel,
    SeqRecModelConfigFactory,
    SeqRecModelFactory,
    SeqRecOutputFactory,
)
from .sasrec import SASRecModelConfig, SASRecModelOutput


__all__ = [
    "SASRecMOJITOModel",
    "SASRecMOJITOModelConfig",
    "SASRecMOJITOModelOutput",
]


@SeqRecModelConfigFactory.register("sasrec_mojito")
class SASRecMOJITOModelConfig(SASRecModelConfig):
    """Configuration class for SASRecMOJITO model, which extends the base `SeqRecModelConfig`."""

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


@SeqRecOutputFactory.register("sasrec_mojito")
@dataclass
class SASRecMOJITOModelOutput(SASRecModelOutput):
    """Output class for SASRecMOJITO model.

    The `SASRecMOJITOModelOutput` extends the base `SeqRecOutput` without adding any additional attributes.
    """

    adaptive_last_hidden_state: Optional[Float[torch.Tensor, "B L d"]] = None


@SeqRecModelFactory.register("sasrec_mojito")
class SASRecMOJITOModel(SeqRecModel[SASRecMOJITOModelConfig, SASRecMOJITOModelOutput]):
    """SASRecMOJITO model implementation.

    Here we implement a more advanced version of the SASRecMOJITO model using: (1) RoPE
    for positional embeddings, (2) Pre-norm architecture with RMSNorm, (3) SwiGLU
    for feed-forward networks, and (4) 4x intermediate size in feed-forward networks.
    The overall architecture follows the original SASRec design and utilizes the
    implementations in Llama model.

    References:
        - Self-Attentive Sequential Recommendation. ICDM '18.
    """

    config_class = SASRecMOJITOModelConfig

    def __init__(self, config: SASRecMOJITOModelConfig) -> None:
        """Initializes SASRecMOJITO model with the given configuration."""
        super().__init__(config)
        self.config: SASRecMOJITOModelConfig

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

        self.time_emb = MOJITOTimeEmbedding(num_buckets=config.num_buckets, time_emb_dim=config.hidden_size)
        self.time_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.temporal_mix = GaussianMixtureTemporalEmbedding(embedding_dim=config.hidden_size)
        self.adaptive_attention = AdaptiveAttention(
            embedding_dim=config.hidden_size, lambda_trans_seq=config.lambda_trans_seq
        )
        self.num_fism_items = config.num_fism_items  # number of FISM elements to sample for each user

    def forward(
        self,
        input_ids: Int[torch.Tensor, "B L"],
        attention_mask: Int[torch.Tensor, "B L"],
        output_model_loss: bool = False,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        **kwargs,
    ) -> SASRecMOJITOModelOutput:
        """Forward pass for SASRecMOJITO model.

        Args:
            input_ids (Int[torch.Tensor, "B L"]): Input item ID sequences of shape (batch_size, seq_len).
            attention_mask (Int[torch.Tensor, "B L"]): Attention masks of shape (batch_size, seq_len).
            output_model_loss (bool): Whether to compute and return the model-specific loss. Default is False.
            output_hidden_states (bool): Whether to return hidden states from all layers. Default is False.
            output_attentions (bool): Whether to return attention weights from all layers. Default is False.
            **kwargs (Any): Additional keyword arguments for the model.

        Returns:
            SASRecMOJITOModelOutput: Model outputs packaged as a `SASRecMOJITOModelOutput` descendant.
        """

        timestamps: Optional[Int[torch.Tensor, "B L"]] = kwargs.pop("timestamps", None)
        if timestamps is None:  # pragma: no cover - defensive check
            raise ValueError("MOJITOModel.forward requires `timestamps` to be provided via kwargs.")
        # Process timestamps to obtain temporal
        time_embeddings: Float[torch.Tensor, "B L d"] = self.time_emb(timestamps)

        hidden_states: Float[torch.Tensor, "B L d"]
        # hidden_states = self.embed_tokens(input_ids) + self.time_proj(time_embeddings)
        hidden_states = self.temporal_mix(
            item_emb=self.embed_tokens(input_ids),
            time_emb=self.time_proj(time_embeddings),
        )

        causal_mask: Float[torch.Tensor, "B 1 L L"]
        causal_mask = create_attention_mask(attention_mask, is_causal=True)

        position_embeddings: Tuple[Float[torch.Tensor, "B L head_dim"], Float[torch.Tensor, "B L head_dim"]]
        position_embeddings = self.rotary_emb(hidden_states)

        # get global_emb
        user_fism_items: Int[torch.Tensor, "B N"] = sample_user_fism_items(
            input_ids, self.num_fism_items, pad_token_id=0
        )
        adaptive_last_hidden_state: Optional[Float[torch.Tensor, "B L d"]] = None
        adaptive_last_hidden_state = self.adaptive_attention(
            seq=hidden_states,
            user_fism_items=self.embed_tokens(user_fism_items),
        )

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

        return SASRecMOJITOModelOutput(
            last_hidden_state=hidden_states,
            model_loss=model_loss if output_model_loss else None,
            hidden_states=tuple(all_hidden_states) if output_hidden_states else None,
            attentions=tuple(all_attentions) if output_attentions else None,
            adaptive_last_hidden_state=adaptive_last_hidden_state,
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


class FISMAttention(nn.Module):
    """
    Factorized Item Similarity Model (FISM) Adaptive Attention Vector
    This module computes attention over item sequences based on a user-defined FISM mechanism.
    """

    def __init__(self, beta=1.0):
        """
        Args:
            beta (float, optional): A hyperparameter to scale the softmax weight normalization. Default is 1.0.
        """
        super().__init__()
        self.beta = beta

    def forward(self, seq: torch.Tensor, fism_items: torch.Tensor) -> torch.Tensor:
        """
        Args:
            seq: The embedding of the input sequence [B, T, D], where:
                 B = batch size, T = sequence length, D = embedding dimension.
            fism_items: Factorized item embeddings [B, M, D], where:
                 M = number of FISM-related items to calculate attention with.

        Returns:
            att_vecs: Attention-weighted item vectors for the sequence [B, T, D].
        """
        # Calculate dot-product similarity
        w_ij = torch.matmul(seq, fism_items.transpose(-1, -2))  # Shape: [B, T, M]

        # Compute exponential weights
        exp_wij = torch.exp(w_ij)  # Shape: [B, T, M]
        exp_sum = torch.sum(exp_wij, dim=-1, keepdim=True)  # [B, T, 1]

        # Apply beta scaling (if beta != 1.0, scale the weights)
        if self.beta != 1.0:
            exp_sum = exp_sum**self.beta  # Perform element-wise power

        # Normalize to get attention weights
        att_weights = exp_wij / exp_sum  # Shape: [B, T, M]

        # Compute attention-weighted vectors
        att_vecs = torch.matmul(att_weights, fism_items)  # Shape: [B, T, D]
        return att_vecs


class AdaptiveAttention(nn.Module):
    """
    Adaptive Attentive Sequence Layer.

    This layer combines FISM's attention-weighted vectors with the input sequence
    based on the lambda_trans_seq parameter.
    """

    def __init__(self, embedding_dim: int, lambda_trans_seq: float = 0.1):
        """
        Args:
            embedding_dim: Int, the embedding dimension for input sequences
            lambda_trans_seq: float, weight to control blending ratio of sequence-to-context attention
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.lambda_trans_seq = lambda_trans_seq  # Weight for blending

        self.fism_attention = FISMAttention()  # FISM attention module

    def forward(
        self,
        seq: torch.Tensor,  # [B, T, D] Input sequence embeddings
        user_fism_items: torch.Tensor,  # [B, M, D] User-related FISM embedding vectors
    ) -> torch.Tensor:
        """
        Forward propagation for adaptive attention sequence.

        Args:
            seq: Input sequence embedding [B, T, D].
            user_fism_items: User embeddings for items (FISM-related) [B, M, D].

        Returns:
            att_fism_seq: FISM attention-modulated item embeddings [B, T, D].
        """
        # Normalize input
        seq_normalized = F.normalize(seq, p=2, dim=-1)  # Normalize along embedding dimension

        # Compute FISM attention vectors
        att_vectors = self.fism_attention(seq_normalized, user_fism_items)  # [B, T, D]

        # Weighted combination of sequence and attention vectors
        if self.lambda_trans_seq < 1.0:
            att_fism_seq = seq * (1.0 - self.lambda_trans_seq) + (seq * att_vectors) * self.lambda_trans_seq
        else:
            att_fism_seq = seq * att_vectors  # No control; 100% attention blending

        return att_fism_seq


import numpy as np


def sample_user_fism_items(input_ids: torch.Tensor, n_fism_elems: int, pad_token_id: int = 0):
    B, L = input_ids.size()
    device = input_ids.device
    user_fism_items = []

    for i in range(B):
        row = input_ids[i]
        valid = row[row != pad_token_id]
        hist = valid.cpu().numpy()
        hist_len = len(hist)
        if hist_len == 0:
            selected = np.full(n_fism_elems, pad_token_id, dtype=np.int64)
        elif hist_len >= n_fism_elems:
            selected = np.random.choice(hist, n_fism_elems, replace=False)
        else:
            mul = n_fism_elems // hist_len
            res = n_fism_elems % hist_len
            selected = np.concatenate([np.tile(hist, mul), np.random.choice(hist, res, replace=False)])
        user_fism_items.append(torch.from_numpy(selected))
    return torch.stack(user_fism_items, dim=0).to(device)
