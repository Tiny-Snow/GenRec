import torch
import torch.nn as nn

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


def test_feedforward_network_respects_custom_activation() -> None:
    hidden_size = 2
    intermediate_size = 3
    ffn = FeedForwardNetwork(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        ffn_bias=True,
        activation=nn.Identity(),
    )

    with torch.no_grad():
        ffn.fc1.weight.fill_(1.0)
        ffn.fc1.bias.zero_()
        ffn.fc2.weight.fill_(1.0)
        ffn.fc2.bias.zero_()

    inputs = torch.tensor([[[1.0, 2.0]]])
    outputs = ffn(inputs)

    # fc1: sum to 3 along hidden dims, replicated across intermediate_size; fc2: sums three 3's to 9 for each dim
    expected = torch.full_like(inputs, 9.0)
    torch.testing.assert_close(outputs, expected)
