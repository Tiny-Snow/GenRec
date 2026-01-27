"""Basic RQ-VAE Trainer for quantizer models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, TypeVar, Union

from jaxtyping import Float
import torch

from ...models import QuantizerModel, QuantizerOutput
from .base import (
    QuantizerTrainer,
    QuantizerTrainerFactory,
    QuantizerTrainingArguments,
    QuantizerTrainingArgumentsFactory,
)

__all__ = [
    "RQVAEQuantizerTrainer",
    "RQVAEQuantizerTrainingArguments",
]


_RQVAEModel = TypeVar("_RQVAEModel", bound="QuantizerModel[Any, Any]")


@QuantizerTrainingArgumentsFactory.register("rqvae")
@dataclass
class RQVAEQuantizerTrainingArguments(QuantizerTrainingArguments):
    """Training arguments for `RQVAEQuantizerTrainer`."""

    pass


@QuantizerTrainerFactory.register("rqvae")
class RQVAEQuantizerTrainer(QuantizerTrainer[_RQVAEModel, RQVAEQuantizerTrainingArguments]):
    """Basic RQ-VAE Trainer for quantizer models.

    This trainer extends the base `QuantizerTrainer` to implement the loss function
    specific to RQ-VAE models. No additional training arguments are required beyond
    those provided by the base class.
    """

    args: RQVAEQuantizerTrainingArguments
    model: _RQVAEModel

    def compute_quantizer_loss(
        self,
        inputs: dict[str, Union[torch.Tensor, Any]],
        outputs: QuantizerOutput,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ) -> Float[torch.Tensor, ""]:
        """Computes the model-agnostic quantizer loss for a batch of inputs and model outputs.

        This method should be implemented by all subclasses to compute the quantizer-specific loss
        components, e.g., reconstruction loss, codebook loss, and commitment loss.

        Args:
            inputs (dict[str, Union[torch.Tensor, Any]]): Dictionary of input tensors, i.e., the
                `QuantizerCollator` output.
            outputs (QuantizerOutput): Output from the quantizer model's forward pass.
            num_items_in_batch (Optional[torch.Tensor]): Optional tensor indicating the number of
                valid items in each sequence in the batch (excluding padding).

        Returns:
            Float[torch.Tensor, ""]: Scalar tensor representing the computed quantizer loss.
        """
        assert outputs.reconstruction_loss is not None, "Reconstruction loss must be provided in outputs."
        assert outputs.codebook_loss is not None, "Codebook loss must be provided in outputs."
        assert outputs.commitment_loss is not None, "Commitment loss must be provided in outputs."

        reconstruction_loss: Float[torch.Tensor, "B"] = outputs.reconstruction_loss
        codebook_loss: Float[torch.Tensor, "B"] = outputs.codebook_loss
        commitment_loss: Float[torch.Tensor, "B"] = outputs.commitment_loss

        loss = (
            reconstruction_loss.mean()
            + self.args.codebook_loss_weight * codebook_loss.mean()
            + self.args.commitment_loss_weight * commitment_loss.mean()
        )
        return loss
