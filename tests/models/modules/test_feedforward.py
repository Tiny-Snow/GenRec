import torch

from genrec.models.modules.feedforward import SwiGLU


def test_swiglu_returns_hidden_sized_outputs() -> None:
    hidden_size = 8
    intermediate_size = 16
    swiglu = SwiGLU(hidden_size=hidden_size, intermediate_size=intermediate_size)

    inputs = torch.randn(2, 5, hidden_size)
    outputs = swiglu(inputs)

    assert outputs.shape == inputs.shape
