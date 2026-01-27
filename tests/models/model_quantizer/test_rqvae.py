import numpy as np
import torch
import torch.nn as nn

from genrec.models.model_quantizer.rqvae import RQVAEModel, RQVAEModelConfig


def test_rqvae_forward_outputs_and_losses() -> None:
    config = RQVAEModelConfig(
        embed_dim=12,
        hidden_sizes=(8, 4),
        num_codebooks=2,
        codebook_size=16,
        codebook_dim=6,
        kmeans_init=False,
    )
    model = RQVAEModel(config)

    batch_size = 3
    item_id = torch.arange(1, batch_size + 1)
    item_embedding = torch.randn(batch_size, config.embed_dim)

    output = model(
        item_id=item_id,
        item_embedding=item_embedding,
        output_loss=True,
        output_model_loss=True,
        output_embeddings=True,
    )

    assert output.semantic_ids.shape == (batch_size, config.num_codebooks)
    assert output.quantized_embeddings is not None
    assert output.quantized_embeddings.shape == (batch_size, config.num_codebooks, config.codebook_dim)
    assert output.residual_embeddings is not None
    assert output.residual_embeddings.shape == (batch_size, config.num_codebooks, config.codebook_dim)
    assert output.decoded_embeddings is not None
    assert output.decoded_embeddings.shape == (batch_size, config.embed_dim)
    assert output.reconstruction_loss is not None
    assert output.reconstruction_loss.shape == (batch_size,)
    assert output.codebook_loss is not None
    assert output.codebook_loss.shape == (batch_size,)
    assert output.commitment_loss is not None
    assert output.commitment_loss.shape == (batch_size,)
    assert output.model_loss is None


def test_rqvae_forward_without_optional_embeddings() -> None:
    config = RQVAEModelConfig(
        embed_dim=10,
        hidden_sizes=(6,),
        num_codebooks=1,
        codebook_size=8,
        codebook_dim=4,
        kmeans_init=False,
    )
    model = RQVAEModel(config)

    batch_size = 2
    item_id = torch.arange(1, batch_size + 1)
    item_embedding = torch.randn(batch_size, config.embed_dim)

    output = model(
        item_id=item_id,
        item_embedding=item_embedding,
        output_loss=True,
        output_embeddings=False,
    )

    assert output.quantized_embeddings is None
    assert output.residual_embeddings is None
    assert output.decoded_embeddings is None
    assert output.reconstruction_loss is not None


def test_rqvae_quantize_returns_expected_losses() -> None:
    config = RQVAEModelConfig(
        embed_dim=2,
        hidden_sizes=(2,),
        num_codebooks=1,
        codebook_size=2,
        codebook_dim=2,
        kmeans_init=False,
    )
    model = RQVAEModel(config)

    embeddings = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 1.0],
        ],
        requires_grad=True,
    )
    codebook = torch.tensor(
        [
            [0.0, 0.0],
            [0.5, 0.5],
        ],
        dtype=torch.float32,
    )

    ids, quantized, codebook_loss, commitment_loss = model._quantize(embeddings, codebook)

    assert ids.tolist() == [0, 1]
    torch.testing.assert_close(quantized, torch.tensor([[0.0, 0.0], [0.5, 0.5]]))
    assert codebook_loss.shape == (2,)
    assert commitment_loss.shape == (2,)

    quantized.sum().backward()
    assert embeddings.grad is not None


def test_rqvae_initialize_codebooks_uses_kmeans(monkeypatch) -> None:
    config = RQVAEModelConfig(
        embed_dim=6,
        hidden_sizes=(4,),
        num_codebooks=2,
        codebook_size=2,
        codebook_dim=3,
        kmeans_init=True,
        kmeans_max_iter=1,
    )
    model = RQVAEModel(config)

    class DummyKMeans:
        call_count = 0

        def __init__(self, n_clusters, **kwargs):
            self.n_clusters = n_clusters

        def fit(self, residual):
            DummyKMeans.call_count += 1
            fill = float(DummyKMeans.call_count)
            self.cluster_centers_ = np.full((self.n_clusters, residual.shape[1]), fill, dtype=np.float32)

    monkeypatch.setattr("genrec.models.model_quantizer.rqvae.KMeans", DummyKMeans)

    item_embeddings = torch.randn(3, config.embed_dim)
    model.initialize_codebooks(item_embeddings)

    for idx, codebook in enumerate(model.codebooks, start=1):
        expected = torch.full((config.codebook_size, config.codebook_dim), float(idx))
        torch.testing.assert_close(codebook.weight.data, expected)


def test_rqvae_initialize_codebooks_skips_when_disabled() -> None:
    config = RQVAEModelConfig(kmeans_init=False)
    model = RQVAEModel(config)

    before = [codebook.weight.clone() for codebook in model.codebooks]
    embeddings = torch.randn(5, config.embed_dim)
    model.initialize_codebooks(embeddings)

    for original, codebook in zip(before, model.codebooks, strict=True):
        torch.testing.assert_close(codebook.weight, original)


def test_rqvae_forward_computes_reconstruction_loss_without_embeddings() -> None:
    config = RQVAEModelConfig(
        embed_dim=2,
        hidden_sizes=(),
        num_codebooks=1,
        codebook_size=1,
        codebook_dim=2,
        kmeans_init=False,
    )
    model = RQVAEModel(config)

    model.encoder = nn.Identity()  # type: ignore[assignment]
    model.decoder = nn.Identity()  # type: ignore[assignment]
    model.codebooks[0].weight.data.zero_()

    item_embedding = torch.tensor([[1.0, -1.0]])
    output = model(
        item_id=torch.tensor([1]),
        item_embedding=item_embedding,
        output_loss=True,
        output_embeddings=False,
    )

    assert output.quantized_embeddings is None
    assert output.residual_embeddings is None
    assert output.decoded_embeddings is None
    assert output.reconstruction_loss is not None
    torch.testing.assert_close(output.reconstruction_loss, torch.tensor([1.0]))


def test_rqvae_forward_skips_decoder_when_not_requested() -> None:
    config = RQVAEModelConfig(
        embed_dim=2,
        hidden_sizes=(),
        num_codebooks=1,
        codebook_size=1,
        codebook_dim=2,
        kmeans_init=False,
    )
    model = RQVAEModel(config)

    class FailingDecoder(nn.Module):
        def forward(self, x):  # pragma: no cover - should not be called
            raise AssertionError("Decoder should not be invoked")

    model.decoder = FailingDecoder()
    model.encoder = nn.Identity()  # type: ignore[assignment]
    model.codebooks[0].weight.data.zero_()

    output = model(
        item_id=torch.tensor([1]),
        item_embedding=torch.zeros(1, config.embed_dim),
        output_loss=False,
        output_embeddings=False,
    )

    assert output.decoded_embeddings is None
    assert output.reconstruction_loss is None
    assert output.codebook_loss is None
    assert output.commitment_loss is None


def test_rqvae_forward_invokes_decoder_when_loss_requested() -> None:
    config = RQVAEModelConfig(
        embed_dim=3,
        hidden_sizes=(),
        num_codebooks=1,
        codebook_size=1,
        codebook_dim=3,
        kmeans_init=False,
    )
    model = RQVAEModel(config)

    class RecordingDecoder(nn.Module):
        def __init__(self, output_dim: int) -> None:
            super().__init__()
            self.called = False
            self.output_dim = output_dim

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self.called = True
            return torch.ones(x.size(0), self.output_dim, device=x.device)

    recording_decoder = RecordingDecoder(output_dim=config.embed_dim)
    model.decoder = recording_decoder
    model.encoder = nn.Identity()  # type: ignore[assignment]
    model.codebooks[0].weight.data.zero_()

    output = model(
        item_id=torch.tensor([1]),
        item_embedding=torch.zeros(1, config.embed_dim),
        output_loss=True,
        output_embeddings=False,
    )

    assert recording_decoder.called is True
    assert output.reconstruction_loss is not None
