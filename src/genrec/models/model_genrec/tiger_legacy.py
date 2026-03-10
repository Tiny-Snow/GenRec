"""GenRec Model: TIGER (Legacy)."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Optional, Union

from jaxtyping import Float, Int
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.cache_utils import Cache, EncoderDecoderCache, DynamicCache
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPastAndCrossAttentions, Seq2SeqLMOutput
from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.t5.modeling_t5 import T5PreTrainedModel, T5Stack

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
    "TIGERLegacyModel",
    "TIGERLegacyModelConfig",
    "TIGERLegacyModelOutput",
]


@GenRecModelConfigFactory.register("tiger_legacy")
class TIGERLegacyModelConfig(GenRecModelConfig):
    """Configuration class for TIGER model, which extends the base `GenRecModelConfig`."""

    def __init__(
        self,
        num_heads: int = 4,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
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
            num_heads (int): Number of attention heads. Default is 4.
            num_encoder_layers (int): Number of layers in the encoder. Default is 4.
            num_decoder_layers (int): Number of layers in the decoder. Default is 4.
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
        self.num_hidden_layers = num_decoder_layers  # for KV-cache
        self.linear_dropout = linear_dropout
        self.attention_dropout = attention_dropout
        self.attention_bias = attention_bias
        self.ffn_bias = ffn_bias
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.enable_rope = enable_rope


@GenRecOutputFactory.register("tiger_legacy")
@dataclass
class TIGERLegacyModelOutput(GenRecOutput):
    """Output class for TIGER model.

    The `TIGERModelOutput` extends the base `GenRecModelOutput` without adding new fields.
    """

    pass


@GenRecModelFactory.register("tiger_legacy")
class TIGERLegacyModel(  # type: ignore - T5 legacy implementation
    GenRecModel[TIGERLegacyModelConfig, TIGERLegacyModelOutput, BaseModelOutputWithPastAndCrossAttentions],
    T5PreTrainedModel,
):
    """TIGER model implementation (legacy), following the `T5ForConditionalGeneration` architecture from HuggingFace Transformers.
    This model can be viewed as a base implementation for generative recommendation tasks.

    References:
    - Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. JMLR '20.
    - Recommender Systems with Generative Retrieval. NeurIPS '23.
    """

    _keys_to_ignore_on_load_unexpected = ["decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight"]
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"]

    config_class = TIGERLegacyModelConfig
    output_class = TIGERLegacyModelOutput
    supports_gradient_checkpointing = True

    def __init__(self, config: TIGERLegacyModelConfig) -> None:  # type: ignore - T5 legacy implementation
        """Initializes the TIGER model with the given configuration."""

        config: T5Config = T5Config(
            vocab_size=config.vocab_size,
            # d_model=config.hidden_size,
            # d_kv=config.head_dim,
            # d_ff=config.hidden_size * 4,
            d_model=128,
            d_kv=64,
            d_ff=1024,
            num_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            # num_heads=config.num_heads,
            num_heads=6,
            relative_attention_num_buckets=config.relative_attention_num_buckets,
            relative_attention_max_distance=config.relative_attention_max_distance,
            dropout_rate=config.linear_dropout,
            pad_token_id=config.pad_token_id,  # type: ignore
            eos_token_id=config.eos_token_id,  # type: ignore
            decoder_start_token_id=config.decoder_start_token_id,  # type: ignore
        )
        super().__init__(config)  # type: ignore - T5 legacy implementation

        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)

        # Set up encoder configuration
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.tie_encoder_decoder = False
        self._encoder = T5Stack(encoder_config, self.shared)

        # Set up decoder configuration
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.tie_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self._decoder = T5Stack(decoder_config, self.shared)

        self.gradient_checkpointing = False  # disable gradient checkpointing by default
        self.post_init()  # use PretrainedModel's default weight initialization

    def _tie_weights(self) -> None:
        """Ties the weights of the encoder and decoder embeddings to the shared embeddings if needed."""
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.decoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.lm_head, self.shared)

    @property
    def encoder(self) -> T5Stack:
        """Returns the encoder module."""
        return self._encoder

    @property
    def decoder(self) -> T5Stack:
        """Returns the decoder module."""
        return self._decoder

    def forward(
        self,
        input_ids: Optional[Int[torch.Tensor, "B L_enc"]] = None,
        attention_mask: Optional[Int[torch.Tensor, "B L_enc"]] = None,
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
    ) -> TIGERLegacyModelOutput:
        """Defines the forward pass of TIGERModel. We directly follow the standard implementation in the
        base `GenRecModel`.

        Args:
            input_ids (Optional[Int[torch.Tensor, "B L_enc"]]): Input token sequences of shape (batch_size, seq_len).
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
        # outputs = super().forward(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     decoder_input_ids=decoder_input_ids,
        #     decoder_attention_mask=decoder_attention_mask,
        #     encoder_outputs=encoder_outputs,
        #     past_key_values=past_key_values,
        #     inputs_embeds=inputs_embeds,
        #     decoder_inputs_embeds=decoder_inputs_embeds,
        #     labels=labels,
        #     cache_position=cache_position,
        #     use_cache=use_cache,
        #     output_hidden_states=output_hidden_states,
        #     output_attentions=output_attentions,
        #     output_model_loss=output_model_loss,
        #     **kwargs,
        # )

        # TODO: delete debug code
        # if not self.training:
        #     print("input_ids.shape:", input_ids.shape if input_ids is not None else None)
        #     print("input_ids:", input_ids if input_ids is not None else None)
        #     print("attention_mask.shape:", attention_mask.shape if attention_mask is not None else None)
        #     print("attention_mask:", attention_mask if attention_mask is not None else None)
        #     print("decoder_input_ids.shape:", decoder_input_ids.shape if decoder_input_ids is not None else None)
        #     print("decoder_input_ids:", decoder_input_ids if decoder_input_ids is not None else None)
        #     print(
        #         "decoder_attention_mask.shape:",
        #         decoder_attention_mask.shape if decoder_attention_mask is not None else None,
        #     )
        #     print("decoder_attention_mask:", decoder_attention_mask if decoder_attention_mask is not None else None)

        outputs = self.origin_forward(
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
            # output_model_loss=output_model_loss,
            # **kwargs,
        )

        # if self.config.tie_word_embeddings:
        #     # Rescale output before projecting on vocab
        #     outputs.logits = outputs.logits * (self.config.hidden_size**-0.5)

        # if labels is not None:
        #     loss_fact = nn.CrossEntropyLoss(ignore_index=-100)
        #     outputs.loss = loss_fact(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1))  # type: ignore - T5 legacy implementation

        return outputs

    def origin_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[tuple[tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. T5 is a model with relative position embeddings so you
            should be able to pad the inputs on both the right and the left.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for detail.

            [What are input IDs?](../glossary#input-ids)

            To know more on how to prepare `input_ids` for pretraining take a look a [T5 Training](./t5#training).
        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Indices of decoder input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are decoder input IDs?](../glossary#decoder-input-ids)

            T5 uses the `pad_token_id` as the starting token for `decoder_input_ids` generation. If `past_key_values`
            is used, optionally only the last `decoder_input_ids` have to be input (see `past_key_values`).

            To know more on how to prepare `decoder_input_ids` for pretraining take a look at [T5
            Training](./t5#training).
        decoder_attention_mask (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default.
        decoder_head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules in the decoder. Mask values selected in `[0,
            1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        cross_attn_head_mask (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in
            `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, T5ForConditionalGeneration

        >>> tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
        >>> model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")

        >>> # training
        >>> input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
        >>> labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
        >>> outputs = model(input_ids=input_ids, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits

        >>> # inference
        >>> input_ids = tokenizer(
        ...     "summarize: studies have shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> outputs = model.generate(input_ids)
        >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        >>> # studies have shown that owning a dog is good for you.
        ```"""
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        sequence_output = decoder_outputs[0]

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return TIGERLegacyModelOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)
