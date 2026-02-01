"""Dataset for quantizer training tasks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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
        aux_item_embedding: Optional auxiliary dense embedding vector of the item,
            e.g., produced by SeqRec model.
    """

    item_id: int
    item_embedding: Float[np.ndarray, "D"]
    aux_item_embedding: Optional[Float[np.ndarray, "D_aux"]] = None


@RecDatasetFactory.register("quantizer")
class QuantizerDataset(RecDataset[QuantizerExample]):
    """Dataset variant producing training examples for quantizer training."""

    def __init__(
        self,
        interaction_data_path: Union[pd.DataFrame, str, Path],
        textual_data_path: Optional[Union[pd.DataFrame, str, Path]] = None,
        lm_encoder: Optional[LMEncoder] = None,
        aux_item_embeddings: Optional[Float[np.ndarray, "I+1 D_aux"]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialises the dataset and materialises user-level metadata.

        Args:
            interaction_data_path (Union[pd.DataFrame, str, Path]): Pandas DataFrame or path to a
                pickle file containing `UserID` and `ItemID` columns. We assume that the `UserID`
                begins from 0 and that `ItemID` begins from 1, both being contiguous integers. The
                `ItemID` of 0 is reserved for padding.
            textual_data_path (Optional[Union[pd.DataFrame, str, Path]]): Optional DataFrame or
                pickle file with `ItemID` and `Title` columns.
            lm_encoder (Optional[LMEncoder]): Optional encoder used to transform item titles into
                dense embeddings.
            aux_item_embeddings (Optional[Float[np.ndarray, "I+1 D_aux"]]): Optional auxiliary
                item embeddings, e.g., produced by SeqRec model.
            **kwargs (Any): Additional keyword arguments for the dataset.
        """

        self._aux_item_embeddings = aux_item_embeddings

        super().__init__(
            interaction_data_path,
            DatasetSplitLiteral.TRAIN,
            textual_data_path=textual_data_path,
            lm_encoder=lm_encoder,
            **kwargs,
        )

        if aux_item_embeddings is not None:
            assert (
                aux_item_embeddings.shape[0] == self.item_size + 1
            ), "The number of auxiliary item embeddings must equal item_size + 1."

    def _build_examples(self) -> List[QuantizerExample]:
        """Generates training examples for quantizer training."""
        assert self._item_textual_embeddings is not None, "Item embeddings are required for quantizer training."
        examples = [
            QuantizerExample(
                i,
                self._item_textual_embeddings[i],
                self._aux_item_embeddings[i] if self._aux_item_embeddings is not None else None,
            )
            for i in range(1, self.item_size + 1)
        ]
        return examples

    @property
    def aux_item_embeddings(self) -> Optional[Float[np.ndarray, "I+1 D_aux"]]:
        """Exposes the auxiliary item embeddings, if available."""
        return self._aux_item_embeddings

    @property
    def aux_embedding_dim(self) -> Optional[int]:
        """Returns the dimensionality of the auxiliary item embeddings, if available."""
        if self._aux_item_embeddings is None:  # pragma: no cover - embedding absent
            return None
        return self._aux_item_embeddings.shape[1]


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
            aux_item_embedding: `Optional[Float[torch.Tensor, "B D_aux"]]`.
                Auxiliary item dense embeddings, if available.
    """

    def __init__(
        self,
        dataset: QuantizerDataset,
        config: Optional[QuantizerCollatorConfig] = None,
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        """Configures the collator.

        Args:
            dataset (QuantizerDataset): Dataset split from which examples are drawn.
            config (Optional[QuantizerCollatorConfig]): Optional collator configuration instance.
            seed (int): Random seed for the collator's internal RNG.
            **kwargs (Any): Additional keyword arguments for the collator.
        """
        self._config = config or QuantizerCollatorConfig()

        need_pad_keys: Dict[str, type] = {}
        no_pad_keys: Dict[str, type] = {
            "item_id": np.int64,
            "item_embedding": np.float32,
            "aux_item_embedding": np.float32,
        }
        pad_values: Dict[str, np.generic] = {}

        super().__init__(need_pad_keys, no_pad_keys, pad_values, seed, **kwargs)
