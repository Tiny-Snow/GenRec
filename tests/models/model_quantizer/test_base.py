from __future__ import annotations

import torch

from genrec.models.model_quantizer.base import QuantizerModel, QuantizerModelConfig


class TinyQuantizerModel(QuantizerModel[QuantizerModelConfig, None]):
    config_class = QuantizerModelConfig

    def __init__(self, config: QuantizerModelConfig) -> None:
        super().__init__(config)
        self._param = torch.nn.Parameter(torch.zeros(1))

    def forward(self, *args, **kwargs):  # pragma: no cover - not needed for this test
        raise NotImplementedError

    def initialize_codebooks(self, item_embeddings: torch.Tensor, **kwargs) -> None:  # pragma: no cover - unused
        return None


def test_post_process_quantized_ids_offsets_and_anticollision() -> None:
    config = QuantizerModelConfig(
        embed_dim=4,
        hidden_sizes=(2,),
        num_codebooks=2,
        codebook_size=10,
        codebook_dim=2,
    )
    model = TinyQuantizerModel(config)

    semantic_ids = torch.tensor(
        [
            [0, 1],
            [0, 1],
            [2, 3],
        ],
        dtype=torch.long,
    )

    processed = model.post_process_quantized_ids(semantic_ids)

    assert processed.shape == (3, config.num_codebooks + 1)
    torch.testing.assert_close(processed[:, 0], torch.tensor([1, 1, 3]))
    torch.testing.assert_close(processed[:, 1], torch.tensor([12, 12, 14]))
    torch.testing.assert_close(processed[:, 2], torch.tensor([21, 22, 21]))
