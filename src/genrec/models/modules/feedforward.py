"""Feed-Forward Network (FFN) modules."""

from __future__ import annotations

from jaxtyping import Float
import torch
import torch.nn as nn

__all__ = [
    "SwiGLU",
]


class SwiGLU(nn.Module):
    """SwiGLU-based Feed-Forward Network, following `LlamaMLP`'s implementation."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        ffn_bias: bool = False,
    ) -> None:
        """Initializes SwiGLU module.

        Args:
            hidden_size (int): Dimensionality of the input and output.
            intermediate_size (int): Dimensionality of the intermediate layer.
            ffn_bias (bool): Whether to include bias terms in the linear projections. Default is False.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=ffn_bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=ffn_bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=ffn_bias)
        self.act_fn = nn.functional.silu

    def forward(self, x: Float[torch.Tensor, "... d"]) -> Float[torch.Tensor, "... d"]:
        """Forward pass for SwiGLU.

        Args:
            x (Float[torch.Tensor, "... d"]): Input tensor of shape (..., hidden_size).

        Returns:
            Float[torch.Tensor, "... d"]: Output tensor of shape (..., hidden_size).
        """
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
