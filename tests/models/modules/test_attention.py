import torch

from genrec.models.modules.attention import MaskedSelfAttentionWithRoPE
from genrec.models.modules.posemb import RotaryEmbedding
from genrec.models.modules.utils import create_attention_mask


def test_masked_self_attention_with_rope_preserves_shapes() -> None:
    hidden_size = 8
    num_heads = 2
    head_dim = hidden_size // num_heads

    attention = MaskedSelfAttentionWithRoPE(
        hidden_size=hidden_size,
        head_dim=head_dim,
        num_heads=num_heads,
        attention_dropout=0.0,
        attention_bias=False,
    )

    batch_size, seq_len = 2, 4
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    base_mask = torch.ones(batch_size, seq_len, dtype=torch.float32)
    causal_mask = create_attention_mask(base_mask, is_causal=True)

    rotary = RotaryEmbedding(head_dim=head_dim)
    position_embeddings = rotary(torch.zeros(batch_size, seq_len, head_dim))

    attn_output, attn_weights = attention(
        hidden_states,
        attention_mask=causal_mask,
        position_embeddings=position_embeddings,
    )

    assert attn_output.shape == (batch_size, seq_len, hidden_size)
    assert attn_weights.shape == (batch_size, num_heads, seq_len, seq_len)


def test_masked_self_attention_without_rope_or_mask_behaves_well() -> None:
    hidden_size = 6
    num_heads = 3
    head_dim = hidden_size // num_heads

    attention = MaskedSelfAttentionWithRoPE(
        hidden_size=hidden_size,
        head_dim=head_dim,
        num_heads=num_heads,
        attention_dropout=0.0,
        attention_bias=True,
    )
    attention.eval()  # deterministic dropout branch

    batch_size, seq_len = 1, 3
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)

    attn_output, attn_weights = attention(hidden_states)

    assert attn_output.shape == (batch_size, seq_len, hidden_size)
    assert attn_weights.shape == (batch_size, num_heads, seq_len, seq_len)
    row_sums = attn_weights.sum(dim=-1)
    torch.testing.assert_close(row_sums, torch.ones_like(row_sums))
