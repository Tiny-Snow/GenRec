import pytest
import torch

from genrec.models.model_seqrec.hstu_spring import HSTUSpringModel, HSTUSpringModelConfig
from genrec.models.modules.utils import create_attention_mask


def _dummy_inputs(
    config: HSTUSpringModelConfig, batch_size: int = 2, seq_len: int = 5
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids = torch.randint(1, config.item_size + 1, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    timestamps = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    return input_ids, attention_mask, timestamps


def test_hstu_spring_returns_weighted_model_loss(monkeypatch):
    config = HSTUSpringModelConfig(
        item_size=32,
        hidden_size=8,
        num_attention_heads=2,
        num_hidden_layers=2,
        max_seq_len=16,
        spring_attention_weight=0.7,
        spring_emb_weight=0.3,
    )
    model = HSTUSpringModel(config)

    def mock_power_iteration(self, W, name="", eps=1e-12):  # noqa: D401
        return torch.full((), 2.0, device=W.device, dtype=W.dtype)

    def mock_attention_weight_spectral_norm(self, attn_weight, attention_mask):
        return torch.full((), 3.0, device=attn_weight.device, dtype=attn_weight.dtype)

    monkeypatch.setattr(HSTUSpringModel, "_power_iteration", mock_power_iteration)
    monkeypatch.setattr(
        HSTUSpringModel,
        "_attention_weight_spectral_norm",
        mock_attention_weight_spectral_norm,
    )

    input_ids, attention_mask, timestamps = _dummy_inputs(config, batch_size=2, seq_len=3)
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        timestamps=timestamps,
        output_model_loss=True,
        output_hidden_states=True,
        output_attentions=True,
    )

    assert output.model_loss is not None
    assert output.hidden_states is not None and len(output.hidden_states) == config.num_hidden_layers + 1
    assert output.attentions is not None and len(output.attentions) == config.num_hidden_layers

    device = output.model_loss.device
    dtype = output.model_loss.dtype
    attn_inner = torch.tensor(config.num_attention_heads * 3.0 * 4.0, device=device, dtype=dtype)
    attn_component = torch.sqrt(attn_inner) * torch.tensor(2.0, device=device, dtype=dtype)
    per_layer_attention = torch.log1p(attn_component)
    emb_component = torch.log1p(torch.tensor(2.0, device=device, dtype=dtype))
    expected_model_loss = (
        config.spring_attention_weight * per_layer_attention + config.spring_emb_weight * emb_component
    )

    torch.testing.assert_close(output.model_loss, expected_model_loss)


def test_hstu_spring_normalizes_embeddings_before_spectral_norm(monkeypatch):
    config = HSTUSpringModelConfig(
        item_size=8,
        hidden_size=4,
        num_attention_heads=2,
        num_hidden_layers=1,
        norm_embeddings=True,
        spring_attention_weight=0.0,
        spring_ffn_weight=0.0,
        spring_emb_weight=1.0,
    )
    model = HSTUSpringModel(config)

    with torch.no_grad():
        new_weight = torch.arange((config.item_size + 1) * config.hidden_size, dtype=torch.float32)
        new_weight = new_weight.view(config.item_size + 1, config.hidden_size)
        model._item_embed.weight.copy_(new_weight)

    captured_weight: dict[str, torch.Tensor] = {}

    def spy_power_iteration(self, W, name="", eps=1e-12):  # noqa: D401
        value = torch.ones((), device=W.device, dtype=W.dtype)
        if name == "item_embed_weight":
            captured_weight["normalized"] = W.detach().clone()
        return value

    def fake_attention_weight_spectral_norm(self, attn_weight, attention_mask):  # noqa: D401
        return torch.ones((), device=attn_weight.device, dtype=attn_weight.dtype)

    monkeypatch.setattr(HSTUSpringModel, "_power_iteration", spy_power_iteration)
    monkeypatch.setattr(
        HSTUSpringModel,
        "_attention_weight_spectral_norm",
        fake_attention_weight_spectral_norm,
    )

    input_ids, attention_mask, timestamps = _dummy_inputs(config, batch_size=1, seq_len=3)
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        timestamps=timestamps,
        output_model_loss=True,
    )

    assert output.model_loss is not None
    assert "normalized" in captured_weight

    weight = captured_weight["normalized"]
    row_norms = weight.norm(dim=-1)
    nonzero_mask = row_norms > 0
    torch.testing.assert_close(
        row_norms[nonzero_mask],
        torch.ones_like(row_norms[nonzero_mask]),
        atol=1e-5,
        rtol=1e-5,
    )


def test_hstu_spring_skips_model_loss_when_disabled():
    config = HSTUSpringModelConfig(
        item_size=16,
        hidden_size=6,
        num_attention_heads=2,
        num_hidden_layers=1,
    )
    model = HSTUSpringModel(config)

    input_ids, attention_mask, timestamps = _dummy_inputs(config, batch_size=1, seq_len=4)
    output = model(input_ids=input_ids, attention_mask=attention_mask, timestamps=timestamps)

    assert output.model_loss is None
    assert output.last_hidden_state.shape == (1, 4, config.hidden_size)


def test_hstu_spring_requires_timestamps_argument():
    config = HSTUSpringModelConfig(item_size=8, hidden_size=6, num_attention_heads=2, num_hidden_layers=1)
    model = HSTUSpringModel(config)

    input_ids, attention_mask, _ = _dummy_inputs(config, batch_size=1, seq_len=3)

    with pytest.raises(ValueError):
        model(input_ids=input_ids, attention_mask=attention_mask)


def test_hstu_spring_respects_final_layer_norm_flag():
    base_kwargs = dict(item_size=8, hidden_size=6, num_attention_heads=2, num_hidden_layers=1)

    config_ln = HSTUSpringModelConfig(final_layer_norm=True, **base_kwargs)
    model_ln = HSTUSpringModel(config_ln)
    assert isinstance(model_ln.final_layer_norm, torch.nn.Module)
    assert model_ln.final_layer_norm.__class__.__name__ == "RMSNorm"

    config_id = HSTUSpringModelConfig(final_layer_norm=False, **base_kwargs)
    model_id = HSTUSpringModel(config_id)
    assert isinstance(model_id.final_layer_norm, torch.nn.Identity)


def test_hstu_spring_power_iteration_registers_and_updates_buffers():
    config = HSTUSpringModelConfig(
        item_size=8,
        hidden_size=4,
        num_attention_heads=2,
        num_hidden_layers=1,
        spectral_norm_iters=50,
    )
    model = HSTUSpringModel(config)

    torch.manual_seed(0)
    weight_first = torch.randn(3, 3, dtype=torch.float32)
    sigma_first_expected = torch.linalg.svdvals(weight_first)[0]
    sigma_first = model._power_iteration(weight_first, name="probe")
    torch.testing.assert_close(sigma_first, sigma_first_expected, atol=1e-4, rtol=1e-4)
    assert hasattr(model, "probe_u") and hasattr(model, "probe_v")
    first_u = model.probe_u.clone()

    weight_second = torch.randn(3, 3, dtype=torch.float32)
    sigma_second_expected = torch.linalg.svdvals(weight_second)[0]
    sigma_second = model._power_iteration(weight_second, name="probe")
    torch.testing.assert_close(sigma_second, sigma_second_expected, atol=1e-4, rtol=1e-4)
    assert not torch.allclose(model.probe_u, first_u)

    weight_third = torch.tensor([[2.0, 1.0], [1.0, 2.0]], dtype=torch.float32)
    sigma_third = model._power_iteration(weight_third)
    assert not hasattr(model, "_u")
    assert torch.isfinite(sigma_third)


def test_hstu_spring_attention_weight_spectral_norm_masks_padding():
    config = HSTUSpringModelConfig(
        item_size=8,
        hidden_size=4,
        num_attention_heads=2,
        num_hidden_layers=1,
        spring_attention_temperature=0.5,
    )
    model = HSTUSpringModel(config)

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
    mask = create_attention_mask(attention_mask, is_causal=True, mask_value=1).bool().squeeze(1)
    masked_attn = attn_weight.masked_fill(mask, 0.0)
    query_sums = masked_attn.sum(dim=-2).flatten()
    mask_flat = attention_mask.bool().flatten()
    masked_query = query_sums[mask_flat]
    expected = torch.logsumexp(masked_query * tau, dim=0) / tau

    torch.testing.assert_close(result, expected)


def test_hstu_spring_adds_ffn_regularization_when_enabled(monkeypatch):
    config = HSTUSpringModelConfig(
        item_size=8,
        hidden_size=8,
        num_attention_heads=2,
        num_hidden_layers=1,
        max_seq_len=8,
        add_ffn=True,
        spring_attention_weight=0.0,
        spring_emb_weight=0.0,
        spring_ffn_weight=1.25,
    )
    model = HSTUSpringModel(config)

    def fake_power_iteration(self, W, name="", eps=1e-12):  # noqa: D401
        if "ffn_0_w1" in name:
            value = 2.0
        elif "ffn_0_w2" in name:
            value = 3.0
        else:
            value = 0.0
        return torch.full((), value, device=W.device, dtype=W.dtype)

    def fake_attention_weight_spectral_norm(self, attn_weight, attention_mask):  # noqa: D401
        return torch.zeros((), device=attn_weight.device, dtype=attn_weight.dtype)

    monkeypatch.setattr(HSTUSpringModel, "_power_iteration", fake_power_iteration)
    monkeypatch.setattr(HSTUSpringModel, "_attention_weight_spectral_norm", fake_attention_weight_spectral_norm)

    input_ids, attention_mask, timestamps = _dummy_inputs(config, batch_size=1, seq_len=4)
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        timestamps=timestamps,
        output_model_loss=True,
    )

    assert output.model_loss is not None
    expected = (
        torch.log1p(torch.tensor(6.0, device=output.model_loss.device, dtype=output.model_loss.dtype))
        * config.spring_ffn_weight
    )
    torch.testing.assert_close(output.model_loss, expected)
