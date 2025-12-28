import torch

from genrec.models.modules.layernorm import RMSNorm


def test_rmsnorm_preserves_input_shape_and_dtype() -> None:
    hidden_size = 10
    rms_norm = RMSNorm(hidden_size)

    inputs = torch.randn(3, 4, hidden_size, dtype=torch.float32)
    outputs = rms_norm(inputs)

    assert outputs.shape == inputs.shape
    assert outputs.dtype == inputs.dtype
