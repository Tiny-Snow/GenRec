"""Feed-Forward Network (FFN) modules."""

from __future__ import annotations

import copy
from typing import List

from jaxtyping import Float
import torch
import torch.nn as nn

__all__ = [
    "FeedForwardNetwork",
    "MLP",
    "SwiGLU",
]


class FeedForwardNetwork(nn.Module):
    """Simple two-layer Feed-Forward Network with GELU activation."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        ffn_bias: bool = False,
        activation: nn.Module = nn.ReLU(),
        dropout: float = 0.0,
    ) -> None:
        """Initializes FeedForwardNetwork module.

        Args:
            hidden_size (int): Dimensionality of the input and output.
            intermediate_size (int): Dimensionality of the intermediate layer.
            ffn_bias (bool): Whether to include bias terms in the linear projections. Default is False.
            activation (nn.Module): Activation function to use between layers. Default is ReLU.
            dropout (float): Dropout rate to apply after the activation. Default is 0.0.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=ffn_bias)
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=ffn_bias)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Float[torch.Tensor, "... d"]) -> Float[torch.Tensor, "... d"]:
        """Forward pass for FeedForwardNetwork.

        Args:
            x (Float[torch.Tensor, "... d"]): Input tensor of shape (..., hidden_size).

        Returns:
            Float[torch.Tensor, "... d"]: Output tensor of shape (..., hidden_size).
        """
        return self.fc2(self.dropout(self.activation(self.fc1(x))))


class MLP(nn.Module):
    """Multi-Layer Perceptron (MLP)."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        activation: nn.Module = nn.ReLU(),
        ffn_bias: bool = False,
    ) -> None:
        """Initializes MLP module.

        Args:
            input_size (int): Dimensionality of the input.
            hidden_sizes (List[int]): List of hidden layer sizes.
            output_size (int): Dimensionality of the output.
            activation (nn.Module): Activation function to use between layers. Default is ReLU.
            ffn_bias (bool): Whether to include bias terms in the linear projections. Default is False.
        """
        super().__init__()
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1], bias=ffn_bias))
            if i < len(layer_sizes) - 2:
                layers.append(copy.deepcopy(activation))
        self.network = nn.Sequential(*layers)

    def forward(self, x: Float[torch.Tensor, "... d"]) -> Float[torch.Tensor, "... d"]:
        """Forward pass for MLP.

        Args:
            x (Float[torch.Tensor, "... d"]): Input tensor of shape (..., input_size).

        Returns:
            Float[torch.Tensor, "... d"]: Output tensor of shape (..., output_size).
        """
        return self.network(x)


class SwiGLU(nn.Module):
    """SwiGLU-based Feed-Forward Network, following `LlamaMLP`'s implementation."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        ffn_bias: bool = False,
        dropout: float = 0.0,
    ) -> None:
        """Initializes SwiGLU module.

        Args:
            hidden_size (int): Dimensionality of the input and output.
            intermediate_size (int): Dimensionality of the intermediate layer.
            ffn_bias (bool): Whether to include bias terms in the linear projections. Default is False.
            dropout (float): Dropout rate to apply after the activation. Default is 0.0.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=ffn_bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=ffn_bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=ffn_bias)
        self.act_fn = nn.functional.silu
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Float[torch.Tensor, "... d"]) -> Float[torch.Tensor, "... d"]:
        """Forward pass for SwiGLU.

        Args:
            x (Float[torch.Tensor, "... d"]): Input tensor of shape (..., hidden_size).

        Returns:
            Float[torch.Tensor, "... d"]: Output tensor of shape (..., hidden_size).
        """
        return self.down_proj(self.dropout(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))
