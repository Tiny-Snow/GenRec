import torch

from genrec.models.modules.layers import LlamaDecoderLayer
from genrec.models.modules.posemb import RotaryEmbedding
from genrec.models.modules.utils import create_attention_mask


def test_standard_decoder_layer_outputs_expected_shapes() -> None:
    hidden_size = 12
    num_heads = 3
    layer = LlamaDecoderLayer(
        hidden_size=hidden_size,
        num_heads=num_heads,
        intermediate_size=hidden_size * 2,
        attention_dropout=0.0,
        attention_bias=False,
        ffn_bias=False,
    )

    batch_size, seq_len = 2, 5
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    base_mask = torch.ones(batch_size, seq_len, dtype=torch.float32)
    causal_mask = create_attention_mask(base_mask, is_causal=True)

    rotary = RotaryEmbedding(head_dim=layer.head_dim)
    position_embeddings = rotary(torch.zeros(batch_size, seq_len, layer.head_dim))

    outputs, attn_weights = layer(
        hidden_states=hidden_states,
        attention_mask=causal_mask,
        position_embeddings=position_embeddings,
    )

    assert outputs.shape == (batch_size, seq_len, hidden_size)
    assert attn_weights.shape == (batch_size, num_heads, seq_len, seq_len)
