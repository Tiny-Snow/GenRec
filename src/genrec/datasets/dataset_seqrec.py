"""Dataset for sequential recommendation tasks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from jaxtyping import Float, Int

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
from .modules.negative_samplers import NegativeSamplerFactory

__all__ = [
    "SeqRecCollator",
    "SeqRecCollatorConfig",
    "SeqRecDataset",
    "SeqRecExample",
]


@RecExampleFactory.register("seqrec")
@dataclass(slots=True)
class SeqRecExample(RecExample):
    """Container storing a full sequential example for decoder-only models.

    The example pairs a trimmed interaction history with shifted label items
    so that decoder-only style recommenders can learn to generate the continuation.

    Attributes:
        user_id: Identifier of the user to which the example belongs.
        input_ids: Trimmed history presented to the model as context.
        labels: Positives aligned with `input_ids` (shifted by one position) that
            the model should generate.
        timestamps: Timestamps aligned with `input_ids`, in Unix time.
    """

    user_id: int
    input_ids: Int[np.ndarray, "L"]
    labels: Int[np.ndarray, "L"]
    timestamps: Int[np.ndarray, "L"]


@RecDatasetFactory.register("seqrec")
class SeqRecDataset(RecDataset[SeqRecExample]):
    """Dataset variant producing full sequential examples for decoder-only style models."""

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
        super().__init__(
            interaction_data_path,
            split,
            max_seq_length,
            min_seq_length,
            sid_cache,
            textual_data_path,
            lm_encoder,
        )
        # recompute training set item popularity
        self._train_item_popularity = self._compute_train_item_popularity()

    def _build_examples(self) -> List[SeqRecExample]:
        """Generates full sequential examples for decoder-only style models."""
        examples: List[SeqRecExample] = []
        user_ids = np.arange(self.user_size, dtype=np.int64)
        for user_id, items, times in zip(user_ids, self.user_interactions, self.user_interaction_timestamps):
            input_ids, labels, timestamps = self._trim_sequence(items, times)
            if len(input_ids) == 0:  # pragma: no cover - insufficient length
                continue
            examples.append(
                SeqRecExample(
                    user_id=int(user_id),
                    input_ids=input_ids,
                    labels=labels,
                    timestamps=timestamps,
                )
            )
        return examples

    def _trim_sequence(
        self,
        items: Int[np.ndarray, "..."],
        times: Int[np.ndarray, "..."],
    ) -> Tuple[Int[np.ndarray, "..."], Int[np.ndarray, "..."], Int[np.ndarray, "..."]]:
        """Trims the interaction history according to the configured split."""
        seq_len = int(items.shape[0])
        if self._split == "train":
            target_idx = seq_len - 2
        elif self._split == "validation":
            target_idx = seq_len - 1
        else:  # test split
            target_idx = seq_len
        start_idx = max(0, target_idx - self._max_seq_length - 1)
        length = target_idx - start_idx

        if length < self._min_seq_length + 1:  # pragma: no cover - insufficient length
            return (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
            )

        input_ids = items[start_idx : start_idx + length - 1].astype(np.int64, copy=True)
        labels = items[start_idx + 1 : start_idx + length].astype(np.int64, copy=True)
        timestamps = times[start_idx : start_idx + length - 1].astype(np.int64, copy=True)
        return input_ids, labels, timestamps

    def _compute_train_item_popularity(self) -> Int[np.ndarray, "I+1"]:
        """Computes training dataset item popularity (non-normalised), i.e., the
        number of interactions per item, returning an array of shape (num_items + 1,).

        ..note::
            This is different from the base class's `_compute_item_popularity` method,
            which computes popularity based on the entire dataset.
        """
        popularity = np.zeros((self.item_size + 1,), dtype=np.int64)
        for items in self.user_interactions:
            target_idx = items.shape[0] - 2
            start_idx = max(0, target_idx - self._max_seq_length - 1)
            for item_id in items[start_idx:target_idx]:
                popularity[item_id] += 1
        return popularity

    @property
    def train_item_popularity(self) -> Int[np.ndarray, "I+1"]:
        """Returns the item popularity computed from the training set only."""
        return self._train_item_popularity


@RecCollatorConfigFactory.register("seprec")
@dataclass
class SeqRecCollatorConfig(RecCollatorConfig):
    """Runtime settings consumed by `SeqRecCollator`.

    Attributes:
        num_negative_samples: Number of negatives to sample per instance.
        negative_sampling_strategy: Name of the negative sampling strategy to use.
    """

    num_negative_samples: int = 0
    negative_sampling_strategy: str = "uniform"


@RecCollatorFactory.register("seqrec")
class SeqRecCollator(RecCollator[SeqRecExample]):
    """Converts `SeqRecExample` objects into a batch of PyTorch tensors.

    Returns:
        Dict[str, torch.Tensor]: A dictionary mapping string keys to batched tensors, including:

            user_id: `Int[torch.Tensor, "B"]`.
                User IDs.
            input_ids: `Int[torch.Tensor, "B L"]`.
                Input item ID sequences.
            labels: `Int[torch.Tensor, "B L"]`.
                Label item ID sequences.
            timestamps: `Int[torch.Tensor, "B L"]`.
                Input timestamp sequences.
            attention_mask: `Int[torch.Tensor, "B L"]`.
                Attention masks for inputs.
            negative_item_ids: `Optional[Int[torch.Tensor, "B N"]]`.
                Sampled negative item IDs, when negatives are requested.
    """

    def __init__(
        self,
        dataset: SeqRecDataset,
        config: Optional[SeqRecCollatorConfig] = None,
        seed: int = 42,
    ) -> None:
        """Configures the collator.

        Args:
            dataset (SeqRecDataset): Dataset split from which examples are drawn. This is only
                used to initialize the negative sampler to get some global objects. You can also
                use this collator with other splits of the same dataset.
            config (Optional[SeqRecCollatorConfig]): Optional collator configuration instance.
            seed (int): Random seed for the collator's internal RNG.
        """
        self._config = config or SeqRecCollatorConfig()

        assert self._config.num_negative_samples >= 0, "num_negative_samples must be non-negative."
        self._negative_sampler = NegativeSamplerFactory.create(
            self._config.negative_sampling_strategy,
            dataset=dataset,
        )

        self._item_size = dataset.item_size

        need_pad_keys: Dict[str, type] = {
            "input_ids": np.int64,
            "labels": np.int64,
            "timestamps": np.int64,
            "attention_mask": np.int64,
        }
        no_pad_keys: Dict[str, type] = {
            "user_id": np.int64,
        }
        pad_values: Dict[str, np.generic] = {
            "input_ids": self._config.pad_item,
            "labels": self._config.pad_label,
            "timestamps": np.int64(0),
            "attention_mask": np.int64(0),
        }

        super().__init__(need_pad_keys, no_pad_keys, pad_values, seed)

    def _process_before_padding(
        self,
        batch: List[Dict[str, np.ndarray]],
        batch_seed: int,
    ) -> None:
        """Add attention masks to the batch before padding."""
        for sample in batch:
            seq_len = sample["input_ids"].shape[0]
            sample["attention_mask"] = np.ones((seq_len,), dtype=np.int64)

    def _process_after_padding(
        self,
        need_pad_batch: Dict[str, np.ndarray],
        no_pad_batch: Dict[str, np.ndarray],
        batch_seed: int,
    ) -> None:
        """Perform negative sampling after padding, if requested."""
        if self._config.num_negative_samples == 0:  # pragma: no cover - no negatives requested
            return
        user_histories = need_pad_batch["input_ids"]
        negative_item_ids: Int[np.ndarray, "B N"] = self._negative_sampler(
            history=user_histories,
            num_samples=self._config.num_negative_samples,
            batch_seed=batch_seed,
        )
        no_pad_batch["negative_item_ids"] = negative_item_ids
