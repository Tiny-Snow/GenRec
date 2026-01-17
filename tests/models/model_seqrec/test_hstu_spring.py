import pytest
import torch
import torch.nn.functional as F

import genrec.models.model_seqrec.hstu_spring as hstu_spring
import genrec.models.modules.layers as layers_mod
from genrec.models.model_seqrec.hstu_spring import HSTUSpringModel, HSTUSpringModelConfig
from genrec.models.modules.utils import create_attention_mask


def test_hstu_spring_returns_weighted_model_loss(monkeypatch) -> None:
    config = HSTUSpringModelConfig(
        item_size=32,
        hidden_size=4,
        num_attention_heads=2,
        num_hidden_layers=1,
        enable_ffn=True,
        spring_attention_weight=0.6,
        spring_ffn_weight=0.3,
        spring_emb_weight=0.2,
    )
    model = HSTUSpringModel(config)

    power_values = {
        "item_embed_weight": 2.0,
        "attn_wo": 3.0,
        "attn_head_0_wv": 1.5,
        "attn_head_1_wv": 2.0,
        "ffn_w1": 1.2,
        "ffn_w2": 0.8,
    }
    attn_sn_iter = iter([4.0, 5.0])

    def fake_power_iteration(module, weight, name="", spectral_norm_iters=1, eps=1e-12):
        value = power_values.get(name, 1.0)
        return torch.tensor(value, device=weight.device, dtype=weight.dtype)

    def fake_attention_weight_spectral_norm(attn_weight, tau, padding_mask=None):
        value = next(attn_sn_iter)
        return torch.tensor(value, device=attn_weight.device, dtype=attn_weight.dtype)

    monkeypatch.setattr(hstu_spring, "spring_power_iteration", fake_power_iteration)
    monkeypatch.setattr(
        layers_mod,
        "spring_attention_weight_spectral_norm",
        fake_attention_weight_spectral_norm,
    )

    batch_size, seq_len = 2, 3
    input_ids = torch.randint(1, config.item_size + 1, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    timestamps = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)

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

    attn_sn = torch.tensor([4.0, 5.0], device=device, dtype=dtype)
    wv_values = torch.tensor(
        [power_values["attn_head_0_wv"], power_values["attn_head_1_wv"]],
        device=device,
        dtype=dtype,
    )
    attn_Av = (attn_sn * (wv_values**2)).sum()
    spring_loss_attn = torch.log1p(
        torch.sqrt(attn_Av) * torch.tensor(power_values["attn_wo"], device=device, dtype=dtype)
    )

    ffn_loss = torch.log1p(
        torch.tensor(power_values["ffn_w1"], device=device, dtype=dtype)
        * torch.tensor(power_values["ffn_w2"], device=device, dtype=dtype)
    )
    emb_loss = torch.log1p(torch.tensor(power_values["item_embed_weight"], device=device, dtype=dtype))

    expected_model_loss = (
        config.spring_attention_weight * spring_loss_attn
        + config.spring_ffn_weight * ffn_loss
        + config.spring_emb_weight * emb_loss
    )

    torch.testing.assert_close(output.model_loss, expected_model_loss)


def test_hstu_spring_skips_model_loss_when_disabled() -> None:
    config = HSTUSpringModelConfig(
        item_size=16,
        hidden_size=4,
        num_attention_heads=2,
        num_hidden_layers=1,
    )
    model = HSTUSpringModel(config)

    batch_size, seq_len = 1, 3
    input_ids = torch.randint(1, config.item_size + 1, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    timestamps = torch.randint(0, 10, (batch_size, seq_len), dtype=torch.long)

    output = model(input_ids=input_ids, attention_mask=attention_mask, timestamps=timestamps)

    assert output.model_loss is None
    assert output.last_hidden_state.shape == (batch_size, seq_len, config.hidden_size)


def test_hstu_spring_forward_requires_timestamps() -> None:
    config = HSTUSpringModelConfig(
        item_size=8,
        hidden_size=4,
        num_attention_heads=2,
        num_hidden_layers=1,
    )
    model = HSTUSpringModel(config)

    input_ids = torch.randint(1, config.item_size + 1, (1, 2))
    attention_mask = torch.ones_like(input_ids)

    with pytest.raises(ValueError):
        model(input_ids=input_ids, attention_mask=attention_mask, output_model_loss=True)


def test_hstu_spring_handles_disabled_pos_emb_and_uses_rope() -> None:
    config = HSTUSpringModelConfig(
        item_size=20,
        hidden_size=8,
        num_attention_heads=2,
        num_hidden_layers=1,
        enable_input_pos_emb=False,
        enable_learnable_rel_posemb=False,
        enable_final_layernorm=True,
    )
    model = HSTUSpringModel(config)

    assert model.input_pos_emb is None
    assert model.rotary_emb is not None
    assert model.final_layernorm is not None

    batch_size, seq_len = 2, 4
    input_ids = torch.randint(1, config.item_size + 1, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    timestamps = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)

    output = model(input_ids=input_ids, attention_mask=attention_mask, timestamps=timestamps)

    assert output.model_loss is None
    assert output.last_hidden_state.shape == (batch_size, seq_len, config.hidden_size)


def test_hstu_spring_model_loss_without_ffn(monkeypatch) -> None:
    config = HSTUSpringModelConfig(
        item_size=12,
        hidden_size=4,
        num_attention_heads=2,
        num_hidden_layers=1,
        enable_ffn=False,
        spring_attention_weight=0.5,
        spring_ffn_weight=0.8,
        spring_emb_weight=0.3,
    )
    model = HSTUSpringModel(config)

    power_values = {
        "item_embed_weight": 1.8,
        "attn_wo": 2.2,
        "attn_head_0_wv": 1.1,
        "attn_head_1_wv": 0.9,
    }
    attn_sn_iter = iter([1.5, 1.2])

    def fake_power_iteration(module, weight, name="", spectral_norm_iters=1, eps=1e-12):
        return torch.tensor(power_values.get(name, 1.0), device=weight.device, dtype=weight.dtype)

    def fake_attention_weight_spectral_norm(attn_weight, tau, padding_mask=None):
        return torch.tensor(next(attn_sn_iter), device=attn_weight.device, dtype=attn_weight.dtype)

    monkeypatch.setattr(hstu_spring, "spring_power_iteration", fake_power_iteration)
    monkeypatch.setattr(
        layers_mod,
        "spring_attention_weight_spectral_norm",
        fake_attention_weight_spectral_norm,
    )

    input_ids = torch.randint(1, config.item_size + 1, (1, 3))
    attention_mask = torch.ones(1, 3, dtype=torch.long)
    timestamps = torch.arange(3, dtype=torch.long).unsqueeze(0)

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        timestamps=timestamps,
        output_model_loss=True,
        output_hidden_states=False,
        output_attentions=True,
    )

    assert output.model_loss is not None
    assert output.attentions is not None

    device = output.model_loss.device
    dtype = output.model_loss.dtype
    attn_sn = torch.tensor([1.5, 1.2], device=device, dtype=dtype)
    wv_values = torch.tensor(
        [power_values["attn_head_0_wv"], power_values["attn_head_1_wv"]],
        device=device,
        dtype=dtype,
    )
    attn_Av = (attn_sn * (wv_values**2)).sum()
    spring_loss_attn = torch.log1p(
        torch.sqrt(attn_Av) * torch.tensor(power_values["attn_wo"], device=device, dtype=dtype)
    )
    emb_loss = torch.log1p(torch.tensor(power_values["item_embed_weight"], device=device, dtype=dtype))
    expected_model_loss = config.spring_attention_weight * spring_loss_attn + config.spring_emb_weight * emb_loss

    torch.testing.assert_close(output.model_loss, expected_model_loss)


def test_hstu_spring_normalizes_embeddings_for_spring_loss(monkeypatch) -> None:
    config = HSTUSpringModelConfig(
        item_size=10,
        hidden_size=4,
        num_attention_heads=2,
        num_hidden_layers=1,
        norm_embeddings=True,
    )
    model = HSTUSpringModel(config)

    captured = {}

    def fake_power_iteration(module, weight, name="", spectral_norm_iters=1, eps=1e-12):
        if name == "item_embed_weight":
            captured["weight"] = weight.detach().clone()
        return torch.ones((), device=weight.device, dtype=weight.dtype)

    def fake_attention_weight_spectral_norm(attn_weight, tau, padding_mask=None):
        return torch.ones((), device=attn_weight.device, dtype=attn_weight.dtype)

    monkeypatch.setattr(hstu_spring, "spring_power_iteration", fake_power_iteration)
    monkeypatch.setattr(
        layers_mod,
        "spring_attention_weight_spectral_norm",
        fake_attention_weight_spectral_norm,
    )

    input_ids = torch.randint(1, config.item_size + 1, (1, 3))
    attention_mask = torch.ones(1, 3, dtype=torch.long)
    timestamps = torch.arange(3, dtype=torch.long).unsqueeze(0)

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        timestamps=timestamps,
        output_model_loss=True,
    )

    assert output.model_loss is not None
    assert "weight" in captured

    expected = F.normalize(model.item_embed_weight, p=2, dim=-1)
    torch.testing.assert_close(captured["weight"], expected)


def test_hstu_spring_power_iteration_registers_and_updates_buffers() -> None:
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
    sigma_first = layers_mod.spring_power_iteration(
        model,
        weight_first,
        name="probe",
        spectral_norm_iters=config.spectral_norm_iters,
    )
    torch.testing.assert_close(sigma_first, sigma_first_expected, atol=1e-4, rtol=1e-4)
    assert hasattr(model, "probe_u") and hasattr(model, "probe_v")
    first_u = model.probe_u.clone()

    weight_second = torch.randn(3, 3, dtype=torch.float32)
    sigma_second_expected = torch.linalg.svdvals(weight_second)[0]
    sigma_second = layers_mod.spring_power_iteration(
        model,
        weight_second,
        name="probe",
        spectral_norm_iters=config.spectral_norm_iters,
    )
    torch.testing.assert_close(sigma_second, sigma_second_expected, atol=1e-4, rtol=1e-4)
    assert not torch.allclose(model.probe_u, first_u)

    weight_third = torch.tensor([[2.0, 1.0], [1.0, 2.0]], dtype=torch.float32)
    sigma_third = layers_mod.spring_power_iteration(
        model,
        weight_third,
        spectral_norm_iters=config.spectral_norm_iters,
    )
    assert not hasattr(model, "_u")
    assert torch.isfinite(sigma_third)


def test_hstu_spring_attention_weight_spectral_norm_masks_padding() -> None:
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

    result = layers_mod.spring_attention_weight_spectral_norm(
        attn_weight,
        tau=config.spring_attention_temperature,
        padding_mask=attention_mask,
    )

    tau = config.spring_attention_temperature
    mask = create_attention_mask(attention_mask, is_causal=True, mask_value=1).bool().squeeze(1)
    masked_attn = attn_weight.masked_fill(mask, 0.0)
    query_sums = masked_attn.sum(dim=-2).flatten()
    masked = query_sums[attention_mask.bool().flatten()]
    expected = torch.logsumexp(masked * tau, dim=0) / tau

    torch.testing.assert_close(result, expected)


def test_hstu_spring_uses_gradient_checkpointing() -> None:
    config = HSTUSpringModelConfig(
        item_size=14,
        hidden_size=6,
        num_attention_heads=2,
        num_hidden_layers=2,
    )
    model = HSTUSpringModel(config)
    model.gradient_checkpointing_enable()
    model.train()

    batch_size, seq_len = 1, 3
    input_ids = torch.randint(1, config.item_size + 1, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    timestamps = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)

    model(input_ids=input_ids, attention_mask=attention_mask, timestamps=timestamps)

    assert model.gradient_checkpointing is True
    assert all(getattr(layer, "gradient_checkpointing", False) for layer in model.layers)
