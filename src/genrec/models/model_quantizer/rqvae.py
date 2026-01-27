"""Quantizer Model: RQ-VAE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from jaxtyping import Float, Int
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules import MLP
from .base import (
    QuantizerModel,
    QuantizerModelConfig,
    QuantizerModelConfigFactory,
    QuantizerModelFactory,
    QuantizerOutput,
    QuantizerOutputFactory,
)

__all__ = [
    "RQVAEModel",
    "RQVAEModelConfig",
    "RQVAEModelOutput",
]


@QuantizerModelConfigFactory.register("rqvae")
class RQVAEModelConfig(QuantizerModelConfig):
    """Configuration class for RQ-VAE model, which extends the base `QuantizerModelConfig`."""

    def __init__(
        self,
        kmeans_init: bool = True,
        kmeans_max_iter: int = 10,
        **kwargs,
    ) -> None:
        """Initializes the configuration with model hyperparameters.

        Args:
            kmeans_init (bool): Whether to initialize the codebooks using k-means clustering
                on the provided item embeddings. If False, random initialization is used.
            kmeans_max_iter (int): Maximum number of iterations for the k-means algorithm.
            **kwargs (Any): Additional keyword arguments for the base `QuantizerModelConfig`.
        """
        super().__init__(**kwargs)
        self.kmeans_init = kmeans_init
        self.kmeans_max_iter = kmeans_max_iter


@QuantizerOutputFactory.register("rqvae")
@dataclass
class RQVAEModelOutput(QuantizerOutput):
    """Output class for RQ-VAE model.

    The `RQVAEModelOutput` class extends the base `QuantizerOutput` without adding any additional attributes.
    """

    pass


@QuantizerModelFactory.register("rqvae")
class RQVAEModel(QuantizerModel):
    """Residual-Quantized Variational AutoEncoder (RQ-VAE) model implementation.

    Here we implement the RQ-VAE model with Kmeans initialization.

    References:
    - Neural Discrete Representation Learning. NeurIPS '17.
    - Autoregressive Image Generation Using Residual Quantization. CVPR '22.
    - Recommender Systems with Generative Retrieval. NeurIPS '23.
    """

    config_class = RQVAEModelConfig

    def __init__(self, config: RQVAEModelConfig) -> None:
        super().__init__(config)
        self.config: RQVAEModelConfig

        self.codebooks = nn.ModuleList(
            nn.Embedding(
                self.config.codebook_size,
                self.config.codebook_dim,
            )
            for _ in range(config.num_codebooks)
        )
        self.encoder = MLP(
            input_size=self.config.embed_dim,
            hidden_sizes=list(self.config.hidden_sizes),
            output_size=self.config.codebook_dim,
            activation=nn.ReLU(),
            ffn_bias=True,
        )
        self.decoder = MLP(
            input_size=self.config.codebook_dim,
            hidden_sizes=list(reversed(self.config.hidden_sizes)),
            output_size=self.config.embed_dim,
            activation=nn.ReLU(),
            ffn_bias=True,
        )

        self.gradient_checkpointing = False  # disable gradient checkpointing by default
        self.post_init()  # use PretrainedModel's default weight initialization

    @torch.no_grad()
    def initialize_codebooks(
        self,
        item_embeddings: Float[torch.Tensor, "I D"],
        **kwargs,
    ) -> None:
        """Initializes the codebooks using the provided item embeddings.

        Applying k-means clustering to initialize the codebooks if `kmeans_init` is True.

        Args:
            item_embeddings (Float[torch.Tensor, "I D"]): Dense item embeddings used for
                initializing the codebooks.

        .. note::
            This implementation is not identical to the RQ-Kmeans, as the latter do not
            use encoder.
        """
        if not self.config.kmeans_init:
            return  # Use random initialization

        embeddings = self.encoder(item_embeddings).cpu().numpy()

        residual = embeddings.copy()
        for code_idx in range(self.config.num_codebooks):
            kmeans = KMeans(
                n_clusters=self.config.codebook_size,
                n_init='auto',
                max_iter=self.config.kmeans_max_iter,
                random_state=42,
            )
            kmeans.fit(residual)
            centroids = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
            self.codebooks[code_idx].weight.data.copy_(centroids)

            distances = torch.cdist(torch.tensor(residual), centroids, p=2)
            nearest_codes = torch.argmin(distances, dim=1)
            quantized = centroids[nearest_codes].numpy()
            residual = residual - quantized

    def _quantize(
        self,
        embeddings: Float[torch.Tensor, "B D_c"],
        codebook: Float[torch.Tensor, "K_c D_c"],
    ) -> Tuple[
        Int[torch.Tensor, "B"], Float[torch.Tensor, "B D_c"], Float[torch.Tensor, "B"], Float[torch.Tensor, "B"]
    ]:
        """Quantizes the input embeddings using the provided codebook.

        In the default implementation, we select the nearest code from the codebook for
        each input embedding based on Euclidean distance. You may override this method
        in subclasses to implement alternative quantization strategies.

        Args:
            embeddings (Float[torch.Tensor, "B D_c"]): Input embeddings to be quantized.
            codebook (Float[torch.Tensor, "K_c D_c"]): Codebook weight used for quantization.

        Returns:
            Tuple[Int[torch.Tensor, "B"], Float[torch.Tensor, "B D_c"], Float[torch.Tensor, "B"], Float[torch.Tensor, "B"]]:
            A tuple containing:
                - semantic_ids: Indices of the selected codes from the codebook.
                - quantized_embeddings: Quantized embeddings corresponding to the selected codes.
                - codebook_loss: The codebook loss values.
                - commitment_loss: The commitment loss values.
        """
        # Compute pairwise distances between embeddings and codebook, select nearest code
        distances: Float[torch.Tensor, "B K_c"] = torch.cdist(embeddings, codebook, p=2)
        semantic_ids: Int[torch.Tensor, "B"] = torch.argmin(distances, dim=1)
        quantized_embeddings: Float[torch.Tensor, "B D_c"] = codebook[semantic_ids]

        # Compute Codebook loss and Commitment loss
        codebook_loss = F.mse_loss(quantized_embeddings, embeddings.detach(), reduction="none").mean(dim=-1)
        commitment_loss = F.mse_loss(embeddings, quantized_embeddings.detach(), reduction="none").mean(dim=-1)

        # Apply Straight-Through Estimator (STE) trick for backpropagation
        # This makes the gradient of quantized_embeddings equal to that of embeddings
        # Then during backpropagation, the gradients of the decoder input (i.e., sum of quantized embeddings)
        # is directly passed to the encoder output (i.e., 0-th residual)
        quantized_embeddings = embeddings + (quantized_embeddings - embeddings).detach()

        return semantic_ids, quantized_embeddings, codebook_loss, commitment_loss

    def forward(
        self,
        item_id: Int[torch.Tensor, "B"],
        item_embedding: Float[torch.Tensor, "B D"],
        output_loss: bool = False,
        output_model_loss: bool = False,
        output_embeddings: bool = False,
        **kwargs,
    ) -> RQVAEModelOutput:
        """Performs a forward pass through the quantizer model.

        Args:
            item_id (Int[torch.Tensor, "B"]): Item IDs corresponding to the input embeddings.
            item_embedding (Float[torch.Tensor, "B D"]): Dense item embeddings to be quantized.
            output_loss (bool): Whether to compute and return the reconstruction and commitment losses. Default is False.
            output_model_loss (bool): Whether to compute and return the model-specific loss. Default is False.
            output_embeddings (bool): Whether to return the (quantized, residual, and decoded) embeddings. Default is False.
            **kwargs (Any): Additional keyword arguments for the forward pass.

        Returns:
            RQVAEModelOutput: Model outputs packaged as a `RQVAEModelOutput` object.
        """
        B = item_embedding.shape[0]
        C = self.config.num_codebooks
        D_c = self.config.codebook_dim

        model_loss = None  # By default, RQ-VAE does not compute model loss internally.
        tot_codebook_loss: Float[torch.Tensor, "B"] = torch.zeros(B, device=item_embedding.device)
        tot_commitment_loss: Float[torch.Tensor, "B"] = torch.zeros(B, device=item_embedding.device)
        all_semantic_ids: Int[torch.Tensor, "B C"] = torch.empty(B, C, dtype=torch.long, device=item_embedding.device)
        all_quantized_embeddings: Float[torch.Tensor, "B C D_c"] = torch.empty(B, C, D_c, device=item_embedding.device)
        all_residual_embeddings: Float[torch.Tensor, "B C D_c"] = torch.empty(B, C, D_c, device=item_embedding.device)

        residual = self.encoder(item_embedding)
        accumulated_quantized = torch.zeros_like(residual)

        for code_idx in range(C):
            codebook = self.codebooks[code_idx]
            assert isinstance(codebook, nn.Embedding), "Codebook must be an instance of nn.Embedding."

            if output_embeddings:
                all_residual_embeddings[:, code_idx, :] = residual

            semantic_ids, quantized_embeddings, codebook_loss, commitment_loss = self._quantize(
                residual, codebook.weight
            )
            residual = residual - quantized_embeddings
            accumulated_quantized = accumulated_quantized + quantized_embeddings

            all_semantic_ids[:, code_idx] = semantic_ids
            if output_embeddings:
                all_quantized_embeddings[:, code_idx, :] = quantized_embeddings

            tot_codebook_loss = tot_codebook_loss + codebook_loss
            tot_commitment_loss = tot_commitment_loss + commitment_loss

        decoded_embeddings: Optional[Float[torch.Tensor, "B D"]] = None
        if output_embeddings or output_loss:
            decoded_embeddings = self.decoder(accumulated_quantized)

        tot_codebook_loss = tot_codebook_loss / C
        tot_commitment_loss = tot_commitment_loss / C
        reconstruction_loss = torch.zeros(B, device=item_embedding.device)
        if output_loss:
            assert (
                decoded_embeddings is not None
            ), "Decoded embeddings must be computed to calculate reconstruction loss."
            reconstruction_loss = F.mse_loss(decoded_embeddings, item_embedding, reduction="none").mean(dim=-1)

        return RQVAEModelOutput(
            semantic_ids=all_semantic_ids,
            quantized_embeddings=all_quantized_embeddings if output_embeddings else None,
            residual_embeddings=all_residual_embeddings if output_embeddings else None,
            decoded_embeddings=decoded_embeddings if output_embeddings else None,
            reconstruction_loss=reconstruction_loss if output_loss else None,
            codebook_loss=tot_codebook_loss if output_loss else None,
            commitment_loss=tot_commitment_loss if output_loss else None,
            model_loss=model_loss if output_model_loss else None,
        )
