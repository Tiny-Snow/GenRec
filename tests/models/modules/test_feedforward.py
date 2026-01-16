import torch

from genrec.models.modules.feedforward import FeedForwardNetwork, SwiGLU


def test_swiglu_returns_hidden_sized_outputs() -> None:
    hidden_size = 8
    intermediate_size = 16
    swiglu = SwiGLU(hidden_size=hidden_size, intermediate_size=intermediate_size)

    inputs = torch.randn(2, 5, hidden_size)
    outputs = swiglu(inputs)

    assert outputs.shape == inputs.shape


def test_feedforward_network_preserves_shape_and_supports_bias() -> None:
    hidden_size = 6
    intermediate_size = 10
    ffn = FeedForwardNetwork(hidden_size=hidden_size, intermediate_size=intermediate_size, ffn_bias=True)

    inputs = torch.randn(3, 4, hidden_size, requires_grad=True)
    outputs = ffn(inputs)

    assert outputs.shape == inputs.shape
    outputs.sum().backward()
    assert inputs.grad is not None


def test_swiglu_backward_pass_with_bias() -> None:
    hidden_size = 5
    intermediate_size = 9
    swiglu = SwiGLU(hidden_size=hidden_size, intermediate_size=intermediate_size, ffn_bias=True)

    inputs = torch.randn(2, 3, hidden_size, requires_grad=True)
    outputs = swiglu(inputs)

    assert outputs.shape == inputs.shape
    outputs.mean().backward()
    assert inputs.grad is not None
