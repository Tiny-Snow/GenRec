import torch

from genrec.models.model_seqrec.sasrec_spring import SASRecSpringModel, SASRecSpringModelConfig


def test_sasrec_spring_returns_weighted_model_loss(monkeypatch):
    config = SASRecSpringModelConfig(
        item_size=32,
        hidden_size=4,
        num_attention_heads=2,
        num_hidden_layers=2,
        spring_attention_weight=0.7,
        spring_ffn_weight=0.2,
        spring_emb_weight=0.5,
    )
    model = SASRecSpringModel(config)

    def mock_power_iteration(self, W, name="", eps=1e-12):
        return torch.full((), 2.0, device=W.device, dtype=W.dtype)

    def mock_attention_weight_spectral_norm(self, attn_weight, attention_mask):
        return torch.full((), 3.0, device=attn_weight.device, dtype=attn_weight.dtype)

    monkeypatch.setattr(SASRecSpringModel, "_power_iteration", mock_power_iteration)
    monkeypatch.setattr(
        SASRecSpringModel,
        "_attention_weight_spectral_norm",
        mock_attention_weight_spectral_norm,
    )

    batch_size, seq_len = 2, 3
    input_ids = torch.randint(1, config.item_size + 1, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_model_loss=True,
        output_hidden_states=True,
        output_attentions=True,
    )

    assert output.model_loss is not None
    assert output.hidden_states is not None and len(output.hidden_states) == config.num_hidden_layers + 1
    assert output.attentions is not None and len(output.attentions) == config.num_hidden_layers

    device = output.model_loss.device
    dtype = output.model_loss.dtype
    log2 = torch.log(torch.tensor(2.0, device=device, dtype=dtype))
    log24 = torch.log(torch.tensor(24.0, device=device, dtype=dtype))
    per_layer_attention = 0.5 * log24 + log2
    ffn_component = 2 * log2
    expected_model_loss = (
        config.spring_attention_weight * per_layer_attention
        + config.spring_ffn_weight * ffn_component
        + config.spring_emb_weight * log2
    )

    torch.testing.assert_close(output.model_loss, expected_model_loss)


def test_sasrec_spring_skips_model_loss_when_disabled():
    config = SASRecSpringModelConfig(
        item_size=16,
        hidden_size=4,
        num_attention_heads=1,
        num_hidden_layers=1,
    )
    model = SASRecSpringModel(config)

    input_ids = torch.randint(1, config.item_size + 1, (1, 2))
    attention_mask = torch.ones_like(input_ids)

    output = model(input_ids=input_ids, attention_mask=attention_mask)

    assert output.model_loss is None
    assert output.last_hidden_state.shape == (1, 2, config.hidden_size)


def test_sasrec_spring_power_iteration_registers_and_updates_buffers():
    config = SASRecSpringModelConfig(
        item_size=8,
        hidden_size=4,
        num_attention_heads=2,
        num_hidden_layers=1,
        spectral_norm_iters=10,
    )
    model = SASRecSpringModel(config)

    weight_first = torch.tensor([[3.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    sigma_first = model._power_iteration(weight_first, name="probe")
    torch.testing.assert_close(sigma_first, torch.tensor(3.0), atol=1e-4, rtol=1e-4)
    assert hasattr(model, "probe_u") and hasattr(model, "probe_v")
    first_u = model.probe_u.clone()

    weight_second = torch.tensor([[1.0, 0.0], [0.0, 4.0]], dtype=torch.float32)
    sigma_second = model._power_iteration(weight_second, name="probe")
    torch.testing.assert_close(sigma_second, torch.tensor(4.0), atol=1e-4, rtol=1e-4)
    assert not torch.allclose(model.probe_u, first_u)

    # calling without a name should not create persistent buffers
    weight_third = torch.tensor([[2.0, 1.0], [1.0, 2.0]], dtype=torch.float32)
    sigma_third = model._power_iteration(weight_third)
    assert not hasattr(model, "_u")
    assert torch.isfinite(sigma_third)


def test_sasrec_spring_attention_weight_spectral_norm_masks_padding():
    config = SASRecSpringModelConfig(
        item_size=8,
        hidden_size=4,
        num_attention_heads=2,
        num_hidden_layers=1,
        spring_attention_temperature=0.5,
    )
    model = SASRecSpringModel(config)

    attn_weight = torch.tensor(
        [
            [
                [0.3, 0.2, 0.1],
                [0.1, 0.4, 0.2],
                [0.2, 0.1, 0.3],
            ]
        ],
        dtype=torch.float32,
    )
    attention_mask = torch.tensor([[1, 0, 1]], dtype=torch.long)

    result = model._attention_weight_spectral_norm(attn_weight, attention_mask)

    tau = config.spring_attention_temperature
    query_sums = attn_weight.sum(dim=-2).flatten()
    masked = query_sums[attention_mask.bool().flatten()]
    expected = torch.logsumexp(masked * tau, dim=0) / tau

    torch.testing.assert_close(result, expected)
