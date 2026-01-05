import torch

from genrec.models.modules.posemb import (
    LearnableInputPositionalEmbedding,
    RelativeBucketedTimeAndPositionAttentionBias,
    RotaryEmbedding,
    apply_rotary_pos_emb,
)


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


def test_learnable_input_positional_embedding_uses_descending_positions() -> None:
    batch_size, seq_len, embed_dim = 1, 3, 2
    module = LearnableInputPositionalEmbedding(
        max_position_embeddings=6,
        embed_dim=embed_dim,
        dropout_rate=0.0,
    )

    with torch.no_grad():
        weights = torch.arange(6 * embed_dim, dtype=torch.float32).view(6, embed_dim)
        module.pos_emb.weight.copy_(weights)

    inputs = torch.zeros(batch_size, seq_len, embed_dim)
    outputs = module(inputs)

    expected_indices = torch.tensor([seq_len - 1 - i for i in range(seq_len)], dtype=torch.long)
    expected = weights[expected_indices]
    torch.testing.assert_close(outputs[0], expected)


def test_learnable_input_positional_embedding_accepts_custom_positions() -> None:
    batch_size, seq_len, embed_dim = 2, 4, 3
    module = LearnableInputPositionalEmbedding(
        max_position_embeddings=10,
        embed_dim=embed_dim,
        dropout_rate=0.0,
    )

    with torch.no_grad():
        weights = torch.arange(10 * embed_dim, dtype=torch.float32).view(10, embed_dim)
        module.pos_emb.weight.copy_(weights)

    inputs = torch.zeros(batch_size, seq_len, embed_dim)
    custom_positions = torch.tensor([[0, 1, 5, 7], [7, 5, 1, 0]], dtype=torch.long)
    outputs = module(inputs, position_ids=custom_positions)

    expected = weights[custom_positions]
    torch.testing.assert_close(outputs, expected)


def test_relative_bucketed_bias_combines_time_and_position_components() -> None:
    module = RelativeBucketedTimeAndPositionAttentionBias(
        max_seq_len=4,
        num_buckets=3,
        bucketization_fn=lambda diffs: diffs.abs().clamp(max=3),
    )

    with torch.no_grad():
        pos_values = torch.arange(module.pos_bias_table.num_embeddings, dtype=torch.float32).unsqueeze(-1)
        time_values = torch.arange(module.time_bias_table.num_embeddings, dtype=torch.float32).unsqueeze(-1)
        module.pos_bias_table.weight.copy_(pos_values)
        module.time_bias_table.weight.copy_(time_values)

    timestamps = torch.tensor([[10, 13, 15]], dtype=torch.long)
    bias = module(timestamps)

    assert bias.shape == (1, 1, 3, 3)

    pos_ids = torch.arange(3)
    rel_pos = pos_ids[None, :] - pos_ids[:, None] + (module.max_seq_len - 1)
    expected_pos = module.pos_bias_table.weight[rel_pos].squeeze(-1)

    time_diffs = timestamps[:, :, None] - timestamps[:, None, :]
    bucketed = module.bucketization_fn(time_diffs)
    bucketed = torch.clamp(bucketed, min=0, max=module.num_buckets).long()
    expected_time = module.time_bias_table.weight[bucketed].squeeze(-1)

    torch.testing.assert_close(bias[:, 0], expected_pos + expected_time)
