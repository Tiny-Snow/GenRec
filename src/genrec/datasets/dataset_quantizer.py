"""Dataset for quantizer training tasks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

from jaxtyping import Float, Int
import numpy as np
import pandas as pd

from .base import (
    DatasetSplitLiteral,
    RecCollator,
    RecCollatorConfig,
    RecCollatorConfigFactory,
    RecCollatorFactory,
    RecDataset,
    RecDatasetFactory,
    RecExample,
    RecExampleFactory,
)
from .modules.lm_encoders import LMEncoder

__all__ = [
    "QuantizerCollator",
    "QuantizerCollatorConfig",
    "QuantizerDataset",
    "QuantizerExample",
]


@RecExampleFactory.register("quantizer")
@dataclass(slots=True)
class QuantizerExample(RecExample):
    """Container storing a single training example for quantizer training.

    Attributes:
        item_id: Identifier of the item.
        item_embedding: Dense embedding vector of the item.
    """

    item_id: int
    item_embedding: Float[np.ndarray, "D"]


@RecDatasetFactory.register("quantizer")
class QuantizerDataset(RecDataset[QuantizerExample]):
    """Dataset variant producing training examples for quantizer training."""

    def __init__(
        self,
        interaction_data_path: Union[pd.DataFrame, str, Path],
        split: DatasetSplitLiteral,
        max_seq_length: int,
        min_seq_length: int,
        sid_cache: Optional[Int[np.ndarray, "I+1 C"]] = None,
        textual_data_path: Optional[Union[pd.DataFrame, str, Path]] = None,
        lm_encoder: Optional[LMEncoder] = None,
    ) -> None:
        assert split == "train", "QuantizerDataset only supports the 'train' split."

        super().__init__(
            interaction_data_path,
            split,
            max_seq_length,
            min_seq_length,
            sid_cache,
            textual_data_path,
            lm_encoder,
        )

    def _build_examples(self) -> List[QuantizerExample]:
        """Generates training examples for quantizer training."""
        assert self._item_embeddings is not None, "Item embeddings are required for quantizer training."
        examples = [QuantizerExample(i, self._item_embeddings[i]) for i in range(1, self.item_size + 1)]
        return examples


@RecCollatorConfigFactory.register("quantizer")
@dataclass
class QuantizerCollatorConfig(RecCollatorConfig):
    """Runtime settings consumed by `QuantizerCollator`."""

    pass


@RecCollatorFactory.register("quantizer")
class QuantizerCollator(RecCollator[QuantizerExample]):
    """Converts `QuantizerExample` objects into a batch of PyTorch tensors.

    Returns:
        Dict[str, torch.Tensor]: A dictionary mapping string keys to batched tensors, including:

            item_id: `Int[torch.Tensor, "B"]`.
                Item IDs.
            item_embedding: `Float[torch.Tensor, "B D"]`.
                Item dense embeddings.
    """

    def __init__(
        self,
        dataset: QuantizerDataset,
        config: Optional[QuantizerCollatorConfig] = None,
        seed: int = 42,
    ) -> None:
        """Configures the collator.

        Args:
            dataset (QuantizerDataset): Dataset split from which examples are drawn.
            config (Optional[QuantizerCollatorConfig]): Optional collator configuration instance.
            seed (int): Random seed for the collator's internal RNG.
        """
        self._config = config or QuantizerCollatorConfig()

        need_pad_keys: Dict[str, type] = {}
        no_pad_keys: Dict[str, type] = {
            "item_id": np.int64,
            "item_embedding": np.float32,
        }
        pad_values: Dict[str, np.generic] = {}

        super().__init__(need_pad_keys, no_pad_keys, pad_values, seed)
