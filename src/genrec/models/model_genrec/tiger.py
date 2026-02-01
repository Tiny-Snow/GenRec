"""GenRec Model: TIGER."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Optional

from jaxtyping import Float, Int
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.cache_utils import Cache, EncoderDecoderCache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

from ..modules import RMSNorm, RotaryEmbedding, T5Block
from ..modules.utils import create_attention_mask
from .base import (
    GenRecModel,
    GenRecModelConfig,
    GenRecModelConfigFactory,
    GenRecModelFactory,
    GenRecOutput,
    GenRecOutputFactory,
    ShiftRightMixin,
)

__all__ = [
    "TIGERModel",
    "TIGERModelConfig",
    "TIGERModelOutput",
]


@GenRecModelConfigFactory.register("tiger")
class TIGERModelConfig(GenRecModelConfig):
    """Configuration class for TIGER model, which extends the base `GenRecModelConfig`."""

    def __init__(
        self,
        num_heads: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        linear_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        attention_bias: bool = False,
        ffn_bias: bool = False,
        relative_attention_num_buckets: int = 32,
        relative_attention_max_distance: int = 128,
        enable_rope: bool = False,
        **kwargs,
    ) -> None:
        """Initializes the configuration with model hyperparameters.

        Args:
            hidden_size (int): Dimensionality of the model's hidden representations.
            num_heads (int): Number of attention heads.
            num_encoder_layers (int): Number of layers in the encoder.
            num_decoder_layers (int): Number of layers in the decoder.
            linear_dropout (float): Dropout rate for the output of attention and feed-forward network. Default is 0.0.
            attention_dropout (float): Dropout rate for attention weights. Default is 0.0.
            attention_bias (bool): Whether to include bias terms in the attention projections. Default is False.
            ffn_bias (bool): Whether to include bias terms in the feed-forward network projections. Default is False.
            relative_attention_num_buckets (int): Number of buckets for relative positional embeddings. Default is 32.
            relative_attention_max_distance (int): Maximum distance for relative positional embeddings. Default is 128.
            enable_rope (bool): Whether to use RoPE instead of learnable relative positional bias. If False, the original
                learnable relative positional bias in T5 will be used. Default is False.
            **kwargs (Any): Additional keyword arguments for the base `GenRecModelConfig`.
        """
        super().__init__(**kwargs)
        self.num_heads = num_heads
        assert self.hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.head_dim = self.hidden_size // num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.linear_dropout = linear_dropout
        self.attention_dropout = attention_dropout
        self.attention_bias = attention_bias
        self.ffn_bias = ffn_bias
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.enable_rope = enable_rope


@GenRecOutputFactory.register("tiger")
@dataclass
class TIGERModelOutput(GenRecOutput):
    """Output class for TIGER model.

    The `TIGERModelOutput` extends the base `GenRecModelOutput` without adding new fields.
    """

    pass


class TIGERStack(PreTrainedModel, ShiftRightMixin[TIGERModelConfig]):
    """Standard T5 stack implementation used in the TIGER model, following HuggingFace's `T5Stack`.

    .. note::
        We do not use the `T5PreTrainedModel._init_weights` method to initialize weights, but directly use
        HuggingFace's default initialization in `PreTrainedModel`.
    """

    config_class = TIGERModelConfig
    supports_gradient_checkpointing = True

    def __init__(self, config: TIGERModelConfig, embed_tokens: nn.Module) -> None:
        """Initializes the T5 stack with the given configuration and token embeddings.

        Args:
            config (TIGERModelConfig): Configuration containing model hyperparameters.
            embed_tokens (nn.Module): Token embedding module.
        """
        super().__init__(config)
        self.config: TIGERModelConfig

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder

        self.num_layers = config.num_decoder_layers if config.is_decoder else config.num_encoder_layers
        self.block = nn.ModuleList(
            [
                T5Block(
                    hidden_size=config.hidden_size,
                    head_dim=config.head_dim,
                    num_heads=config.num_heads,
                    intermediate_size=4 * config.hidden_size,
                    linear_dropout=config.linear_dropout,
                    attention_dropout=config.attention_dropout,
                    attention_bias=config.attention_bias,
                    ffn_bias=config.ffn_bias,
                    is_decoder=config.is_decoder,
                    # only the first layer may compute relative attention bias
                    # the other layers will reuse the same bias
                    has_relative_attention_bias=bool(layer_idx == 0),
                    relative_attention_num_buckets=config.relative_attention_num_buckets,
                    relative_attention_max_distance=config.relative_attention_max_distance,
                    enable_rope=config.enable_rope,
                    layer_idx=layer_idx,
                )
                for layer_idx in range(self.num_layers)
            ]
        )
        self.final_layer_norm = RMSNorm(config.hidden_size, eps=1e-6)
        self.dropout = nn.Dropout(config.dropout_rate)

        self.rotary_emb = RotaryEmbedding(head_dim=config.head_dim)

        self.gradient_checkpointing = False  # disable gradient checkpointing by default
        self.post_init()  # use PretrainedModel's default weight initialization

    def forward(
        self,
        input_ids: Optional[Int[torch.Tensor, "B L"]] = None,
        attention_mask: Optional[Int[torch.Tensor, "B L"]] = None,
        inputs_embeds: Optional[Float[torch.Tensor, "B L d"]] = None,
        encoder_hidden_states: Optional[Float[torch.Tensor, "B L_enc d"]] = None,
        encoder_attention_mask: Optional[Int[torch.Tensor, "B L_enc"]] = None,
        past_key_values: Optional[EncoderDecoderCache] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[Int[torch.Tensor, "L"]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs: Any,
    ) -> BaseModelOutputWithPastAndCrossAttentions:
        """Forward pass for the T5 stack.

        Args:
            input_ids (Optional[Int[torch.Tensor, "B L"]]): Input token sequences of shape (batch_size, seq_len).
                Default is None.
            attention_mask (Optional[Int[torch.Tensor, "B L"]]): Attention masks for inputs of shape
                (batch_size, seq_len). Default is None.
            inputs_embeds (Optional[Float[torch.Tensor, "B L d"]]): Input embeddings of `input_ids` of shape
                (batch_size, seq_len, hidden_size). If provided, `input_ids` will be ignored. Default is None.
            encoder_hidden_states (Optional[Float[torch.Tensor, "B L_enc d"]]): Encoder hidden states for decoder
                stacks of shape (batch_size, seq_len_enc, hidden_size).
            encoder_attention_mask (Optional[Int[torch.Tensor, "B L_enc"]]): Attention masks for encoder hidden states
                of shape (batch_size, seq_len_enc).
            past_key_values (Optional[EncoderDecoderCache]): Past key values for faster decoding. Default is None.
            use_cache (Optional[bool]): Whether to use past key values to speed up decoding. Default is None.
            cache_position (Optional[Int[torch.Tensor, "L"]]): Positions for caching in decoding of shape (seq_len,).
                Default is None.
            output_attentions (Optional[bool]): Whether to return attention weights. Default is None.
            output_hidden_states (Optional[bool]): Whether to return hidden states. Default is None.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            BaseModelOutputWithPastAndCrossAttentions: Model outputs including hidden states, attentions,
                and past key values.
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # Prepare inputs
        if inputs_embeds is None:
            assert input_ids is not None, "Either input_ids or inputs_embeds must be provided."
            inputs_embeds = self.embed_tokens(input_ids)

        assert inputs_embeds is not None
        batch_size, seq_length = inputs_embeds.size()[:2]

        # If gradient checkpointing is enabled, disable use_cache
        if self.gradient_checkpointing and self.training:
            use_cache = False

        # If this module is not a decoder but use_cache is True, raise an error
        if not self.is_decoder and use_cache:
            raise ValueError("`use_cache=True` is only supported for decoder stacks.")

        # Prepare cache for decoder
        if self.is_decoder:
            if use_cache and past_key_values is None:
                past_key_values = EncoderDecoderCache(
                    DynamicCache(config=self.config), DynamicCache(config=self.config)
                )
        else:
            past_key_values = None

        past_key_values_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        if cache_position is None:
            cache_position = torch.arange(
                past_key_values_length, past_key_values_length + seq_length, device=inputs_embeds.device
            )

        # Prepare attention mask for inputs
        if attention_mask is None:
            mask_seq_len = past_key_values_length + seq_length
            attention_mask = torch.ones((batch_size, mask_seq_len), device=inputs_embeds.device)

        if self.is_decoder:
            # for static/compileable cache, we may need to provide kv_seq_len (i.e., max cache size)
            kv_seq_len = None
            if past_key_values is not None and past_key_values.is_compileable:  # pragma: no cover - static cache
                max_cache_shape = past_key_values.get_max_cache_shape()
                if max_cache_shape > 0:
                    kv_seq_len = max_cache_shape

            # decoder causal mask of shape (batch_size, 1, L, L_k)
            # where L_k = max(seq_len, past_key_values_length + seq_len, kv_seq_len or 0)
            causal_mask: Float[torch.Tensor, "B 1 L L_k"]
            causal_mask = create_attention_mask(
                attention_mask,
                tgt_len=seq_length,
                is_causal=True,
                dtype=inputs_embeds.dtype,
                cache_position=cache_position,
                past_key_values_length=past_key_values_length,
                kv_seq_len=kv_seq_len,
            )
        elif attention_mask is not None:
            # encoder attention mask of shape (batch_size, 1, 1, seq_len)
            causal_mask: Float[torch.Tensor, "B 1 1 L"]
            causal_mask = create_attention_mask(
                attention_mask,
                tgt_len=1,
                is_causal=False,
                dtype=inputs_embeds.dtype,
            )
        else:  # pragma: no cover - attention mask is None
            causal_mask = None

        # Process encoder attention mask for decoder, of shape (batch_size, 1, 1, seq_len_enc)
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_seq_len, _ = encoder_hidden_states.size()
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    (encoder_batch_size, encoder_seq_len), device=inputs_embeds.device, dtype=torch.long
                )
            encoder_extended_attention_mask: Optional[Float[torch.Tensor, "B 1 1 L_enc"]]
            encoder_extended_attention_mask = create_attention_mask(
                encoder_attention_mask,
                tgt_len=1,
                is_causal=False,
                dtype=inputs_embeds.dtype,
            )
        else:
            encoder_extended_attention_mask = None

        # Prepare optional outputs, position bias for attention, and RoPE cache
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None

        position_bias = None
        encoder_decoder_position_bias = None

        position_embeddings = None
        if self.config.enable_rope:
            rope_position_ids: Optional[torch.Tensor] = None
            if cache_position is not None and self.is_decoder:
                # cache_position may be 1D (shared across batch) or 2D (per-sample); normalize to (B, L)
                rope_position_ids = cache_position
                if rope_position_ids.dim() == 1:  # pragma: no cover - resize position ids
                    rope_position_ids = rope_position_ids.unsqueeze(0)
                if rope_position_ids.size(0) != batch_size:  # pragma: no cover - resize position ids
                    rope_position_ids = rope_position_ids.expand(batch_size, -1)
            position_embeddings = self.rotary_emb(inputs_embeds, position_ids=rope_position_ids)

        # Model forward pass through each layer
        hidden_states = self.dropout(inputs_embeds)

        for layer_idx, layer in enumerate(self.block):
            # Save current hidden states if needed
            if output_hidden_states:
                assert all_hidden_states is not None
                all_hidden_states = all_hidden_states + (hidden_states,)

            # Model forward, layer outputs is a tuple of:
            # - output hidden states
            # - self attention outputs: position bias, attn weights (if requested)
            # - cross attention outputs: position bias, attn weights (if requested)
            hidden_states, self_attn_outputs, cross_attn_outputs = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_bias=position_bias,
                position_embeddings=position_embeddings,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                past_key_values=past_key_values,
                cache_position=cache_position,
                output_attentions=output_attentions,
            )

            # Share the position bias across layers, save self- and cross-attentions
            position_bias = self_attn_outputs[0]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = cross_attn_outputs[0]

            if output_attentions:
                assert all_attentions is not None
                all_attentions = all_attentions + (self_attn_outputs[1],)

            if output_attentions and self.is_decoder:
                assert all_cross_attentions is not None
                all_cross_attentions = all_cross_attentions + (cross_attn_outputs[1],)

        # Final layer norm
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Save final hidden states if needed
        if output_hidden_states:
            assert all_hidden_states is not None
            all_hidden_states = all_hidden_states + (hidden_states,)

        # Prepare outputs
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )


@GenRecModelFactory.register("tiger")
class TIGERModel(GenRecModel[TIGERModelConfig, TIGERModelOutput, BaseModelOutputWithPastAndCrossAttentions]):
    """TIGER model implementation, following the `T5ForConditionalGeneration` architecture from HuggingFace Transformers.
    This model can be viewed as a base implementation for generative recommendation tasks.

    References:
    - Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. JMLR '20.
    - Recommender Systems with Generative Retrieval. NeurIPS '23.
    """

    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"]

    config_class = TIGERModelConfig
    output_class = TIGERModelOutput
    supports_gradient_checkpointing = True

    def __init__(self, config: TIGERModelConfig) -> None:
        """Initializes the TIGER model with the given configuration."""
        super().__init__(config)
        self.config: TIGERModelConfig

        # Set up encoder configuration
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.tie_encoder_decoder = False
        encoder_config.num_hidden_layers = config.num_encoder_layers
        self._encoder = TIGERStack(encoder_config, self.shared)

        # Set up decoder configuration
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.tie_encoder_decoder = False
        decoder_config.num_hidden_layers = config.num_decoder_layers
        self._decoder = TIGERStack(decoder_config, self.shared)

        self.gradient_checkpointing = False  # disable gradient checkpointing by default
        self.post_init()  # use PretrainedModel's default weight initialization

    def _tie_weights(self) -> None:
        """Ties the weights of the encoder and decoder embeddings to the shared embeddings if needed."""
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.decoder.embed_tokens, self.shared)

    @property
    def encoder(self) -> TIGERStack:
        """Returns the encoder module."""
        return self._encoder

    @property
    def decoder(self) -> TIGERStack:
        """Returns the decoder module."""
        return self._decoder

    def forward(
        self,
        input_ids: Int[torch.Tensor, "B L_enc"],
        attention_mask: Int[torch.Tensor, "B L_enc"],
        decoder_input_ids: Optional[Int[torch.Tensor, "B L_dec"]] = None,
        decoder_attention_mask: Optional[Int[torch.Tensor, "B L_dec"]] = None,
        encoder_outputs: Optional[BaseModelOutputWithPastAndCrossAttentions] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[Float[torch.Tensor, "B L_enc d"]] = None,
        decoder_inputs_embeds: Optional[Float[torch.Tensor, "B L_dec d"]] = None,
        labels: Optional[Int[torch.Tensor, "B L_dec"]] = None,
        cache_position: Optional[Int[torch.Tensor, "#L_dec"]] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_model_loss: Optional[bool] = None,
        **kwargs: Any,
    ) -> TIGERModelOutput:
        """Defines the forward pass of TIGERModel. We directly follow the standard implementation in the
        base `GenRecModel`.

        Args:
            input_ids (Int[torch.Tensor, "B L_enc"]): Input token sequences of shape (batch_size, seq_len).
            attention_mask (Optional[Int[torch.Tensor, "B L_enc"]]): Attention masks for inputs of shape
                (batch_size, seq_len).
            decoder_input_ids (Optional[Int[torch.Tensor, "B L_dec"]]): Decoder input token sequences
                of shape (batch_size, dec_seq_len). If `past_key_values` is used, only the last token
                of `decoder_input_ids` have to be input. Default is None.
            decoder_attention_mask (Optional[Int[torch.Tensor, "B L_dec"]]): Attention masks for decoder inputs
                of shape (batch_size, dec_seq_len). Default is None.
            encoder_outputs (Optional[BaseModelOutputWithPastAndCrossAttentions]): Precomputed encoder outputs.
                Default is None.
            past_key_values (Optional[Cache]): Cached key and value tensors for faster decoding. Default is None.
            inputs_embeds (Optional[Float[torch.Tensor, "B L d"]]): Input embeddings of `input_ids` of shape
                (batch_size, seq_len, hidden_size). If provided, `input_ids` will be ignored. Default is None.
            decoder_inputs_embeds (Optional[Float[torch.Tensor, "B L_dec d"]]): Input embeddings of
                `decoder_input_ids` of shape (batch_size, dec_seq_len, hidden_size). If provided,
                `decoder_input_ids` will be ignored. Default is None.
            labels (Optional[Int[torch.Tensor, "B L_dec"]]): Target token sequences for computin the loss, of
                shape (batch_size, dec_seq_len). Default is None.
            cache_position (Optional[Int[torch.Tensor, "#L_dec"]]): Positions for caching in the decoder.
                Default is None.
            use_cache (Optional[bool]): Whether to use past key values to speed up decoding. Default is None.
            output_attentions (Optional[bool]): Whether to return attention weights. Default is None.
            output_hidden_states (Optional[bool]): Whether to return hidden states. Default is None.
            output_model_loss (Optional[bool]): Whether to compute and return the model-specific loss.
                Default is None.
            **kwargs (Any): Additional keyword arguments for the model.

        Returns:
            TIGERModelOutput: Model outputs packaged as a `GenRecOutput` object.
        """
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            cache_position=cache_position,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            output_model_loss=output_model_loss,
            **kwargs,
        )
