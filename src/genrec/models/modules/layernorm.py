"""Layer Normalization (LN) modules."""

from __future__ import annotations

from jaxtyping import Float
import torch
import torch.nn as nn

__all__ = [
    "RMSNorm",
]


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization, following `LlamaRMSNorm`'s implementation."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        """Initializes RMSNorm module.

        Args:
            hidden_size (int): Dimensionality of the input and output.
            eps (float): Small value to avoid division by zero. Default is 1e-6.
        """
        super().__init__()
        self.weight: Float[torch.Tensor, "d"] = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: Float[torch.Tensor, "... d"]) -> Float[torch.Tensor, "... d"]:
        """Forward pass for RMSNorm.

        Args:
            hidden_states (Float[torch.Tensor, "... d"]): Input tensor of shape (..., hidden_size).

        Returns:
            Float[torch.Tensor, "... d"]: Normalized tensor of the same shape as the input.
        """
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
