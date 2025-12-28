import torch

from genrec.models.modules.posemb import RotaryEmbedding, apply_rotary_pos_emb


def test_rotary_embedding_and_application_shapes() -> None:
    batch_size, seq_len = 2, 4
    head_dim = 4
    num_heads = 2

    rotary = RotaryEmbedding(head_dim=head_dim)
    reference = torch.zeros(batch_size, seq_len, head_dim)
    cos, sin = rotary(reference)

    assert cos.shape == (batch_size, seq_len, head_dim)
    assert sin.shape == (batch_size, seq_len, head_dim)

    query = torch.randn(batch_size, num_heads, seq_len, head_dim)
    key = torch.randn(batch_size, num_heads, seq_len, head_dim)
    rotated_query, rotated_key = apply_rotary_pos_emb(query, key, cos, sin)

    assert rotated_query.shape == query.shape
    assert rotated_key.shape == key.shape


def test_rotary_embedding_respects_custom_position_ids() -> None:
    batch_size, seq_len = 1, 4
    head_dim = 4
    rotary = RotaryEmbedding(head_dim=head_dim)

    reference = torch.zeros(batch_size, seq_len, head_dim)
    custom_positions = torch.tensor([[0, 2, 2, 5]], dtype=torch.long)

    cos_default, sin_default = rotary(reference)
    cos_custom, sin_custom = rotary(reference, position_ids=custom_positions)

    # position 1 in custom tensor matches default position 2 due to explicit id=2
    torch.testing.assert_close(cos_custom[:, 1], cos_default[:, 2])
    torch.testing.assert_close(sin_custom[:, 1], sin_default[:, 2])

    # repeated ids should reuse same embedding row
    torch.testing.assert_close(cos_custom[:, 1], cos_custom[:, 2])
    torch.testing.assert_close(sin_custom[:, 1], sin_custom[:, 2])
