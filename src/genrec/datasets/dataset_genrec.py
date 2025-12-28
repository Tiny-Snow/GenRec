"""Dataset for generative recommendation tasks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

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
    "GenRecCollator",
    "GenRecCollatorConfig",
    "GenRecDataset",
    "GenRecExample",
]


@RecExampleFactory.register("genrec")
@dataclass(slots=True)
class GenRecExample(RecExample):
    """Container storing a single training example for encoder-decoder models.

    The example pairs a truncated interaction history with the next positive
    item that the model is asked to predict.

    Attributes:
        user_id: Identifier of the user to which the example belongs.
        input_ids: Interaction history (already truncated) that conditions the model.
        labels: Next positive item the model is asked to predict.
        timestamps: Timestamps aligned with `input_ids`, in Unix time.
        input_sid_tokens: Optional matrix of SIDs corresponding to `input_ids`.
        target_sid_tokens: Optional SIDs for `labels`.
        input_embeddings: Optional dense embedding matrix aligned with `input_ids`.
        target_embedding: Optional dense embedding vector for `labels`.
    """

    user_id: int
    input_ids: Int[np.ndarray, "L"]
    labels: int
    timestamps: Int[np.ndarray, "L"]
    input_sid_tokens: Optional[Int[np.ndarray, "L C"]] = None
    target_sid_tokens: Optional[Int[np.ndarray, "C"]] = None
    input_embeddings: Optional[Float[np.ndarray, "L D"]] = None
    target_embedding: Optional[Float[np.ndarray, "D"]] = None


@RecDatasetFactory.register("genrec")
class GenRecDataset(RecDataset[GenRecExample]):
    """RecDataset variant producing training pairs for encoder-decoder style models
    using sliding windows over interaction histories.
    """

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

    def _build_examples(self) -> List[GenRecExample]:
        """Generates training pairs for encoder-decoder style models using sliding
        windows over interaction histories.
        """
        examples: List[GenRecExample] = []
        user_ids = np.arange(self.user_size, dtype=np.int64)
        for user_id, items, timestamps in zip(user_ids, self.user_interactions, self.user_interaction_timestamps):
            for context, target, times in self._iter_split(items, timestamps):
                if context.shape[0] < self._min_seq_length:  # pragma: no cover - insufficient length
                    continue
                truncated_context = context[-self._max_seq_length :].astype(np.int64, copy=True)
                truncated_times = times[-self._max_seq_length :].astype(np.int64, copy=True)
                examples.append(
                    self._construct_example(
                        user_id=int(user_id),
                        context=truncated_context,
                        target=target,
                        times=truncated_times,
                    )
                )
        return examples

    def _iter_split(
        self,
        items: Int[np.ndarray, "..."],
        times: Int[np.ndarray, "..."],
    ) -> Iterable[Tuple[Int[np.ndarray, "..."], int, Int[np.ndarray, "..."]]]:
        """Yields (context, target, times) pairs according to the configured split."""
        seq_len = int(items.shape[0])
        if self._split == "train":
            for target_idx in range(1, seq_len - 2):
                yield items[:target_idx], int(items[target_idx]), times[:target_idx]
        elif self._split == "validation":
            if seq_len >= 3:
                yield items[:-2], int(items[-2]), times[:-2]
        else:  # test split
            if seq_len >= 2:
                yield items[:-1], int(items[-1]), times[:-1]

    def _construct_example(
        self,
        user_id: int,
        context: Int[np.ndarray, "..."],
        target: int,
        times: Int[np.ndarray, "..."],
    ) -> GenRecExample:
        """Constructs a GenRecExample from the provided context and target."""
        example = GenRecExample(user_id=user_id, input_ids=context, labels=target, timestamps=times)

        if self._sid_cache is not None:
            example.input_sid_tokens = self._sid_cache[context]
            example.target_sid_tokens = self._sid_cache[target]

        if self._item_embeddings is not None:
            example.input_embeddings = self._item_embeddings[context]
            example.target_embedding = self._item_embeddings[target]

        return example

    def _compute_train_item_popularity(self) -> Int[np.ndarray, "I+1"]:
        """Computes training dataset item popularity (non-normalised), i.e., the
        number of interactions per item, returning an array of shape (num_items + 1,).

        ..note::
            This is different from the base class's `_compute_item_popularity` method,
            which computes popularity based on the entire dataset.
        """
        popularity = np.zeros((self.item_size + 1,), dtype=np.int64)
        for items in self.user_interactions:
            for item_id in items[:-2]:
                popularity[item_id] += 1
        return popularity

    @property
    def train_item_popularity(self) -> Int[np.ndarray, "I+1"]:
        """Returns the item popularity computed from the training set only."""
        return self._train_item_popularity


@RecCollatorConfigFactory.register("genrec")
@dataclass
class GenRecCollatorConfig(RecCollatorConfig):
    """Runtime settings consumed by `GenRecCollator`.

    Attributes:
        pad_sid: Padding value for Semantic ID token.
        num_negative_samples: Number of negatives to sample per instance.
        negative_sampling_strategy: Name of the negative sampling strategy to use.
        need_sid_tokens: Whether to collate SID tokens if present. Note that
            if the dataset examples do not contain SID tokens, this flag should
            be set to `False` to avoid errors.
        need_embeddings: Whether to collate dense embeddings if present. Note that
            if the dataset examples do not contain embeddings, this flag should
            be set to `False` to avoid errors.
    """

    num_negative_samples: int = 0
    negative_sampling_strategy: str = "uniform"
    need_sid_tokens: bool = True
    need_embeddings: bool = False

    @property
    def pad_sid(self) -> np.int64:  # pragma: no cover - default implementation
        return np.int64(0)


@RecCollatorFactory.register("genrec")
class GenRecCollator(RecCollator[GenRecExample]):
    """Converts `GenRecExample` objects into a batch of PyTorch tensors.

    Returns:
        Dict[str, torch.Tensor]: A dictionary mapping string keys to batched tensors, including:

            user_id: `Int[torch.Tensor, "B"]`.
                User IDs.
            input_ids: `Int[torch.Tensor, "B L"]`.
                Input item ID sequences.
            labels: `Int[torch.Tensor, "B"]`.
                Target item IDs.
            timestamps: `Int[torch.Tensor, "B L"]`.
                Input timestamp sequences.
            attention_mask: `Int[torch.Tensor, "B L"]`.
                Attention masks for inputs.
            input_sid_tokens: `Optional[Int[torch.Tensor, "B L C"]]`.
                Input Semantic ID token sequences, when needed.
            target_sid_tokens: `Optional[Int[torch.Tensor, "B C"]]`.
                Target Semantic ID tokens, when needed.
            input_embeddings: `Optional[Float[torch.Tensor, "B L D"]]`.
                Input dense embeddings, when present and needed.
            target_embedding: `Optional[Float[torch.Tensor, "B D"]]`.
                Target dense embeddings, when present and needed.
            negative_item_ids: `Optional[Int[torch.Tensor, "B N"]]`.
                Sampled negative item IDs, when negatives are requested.
            negative_sid_tokens: `Optional[Int[torch.Tensor, "B N C"]]`.
                Sampled negative Semantic ID tokens, when needed.
            negative_embeddings: `Optional[Float[torch.Tensor, "B N D"]]`.
                Sampled negative dense embeddings, when present and needed.
    """

    def __init__(
        self,
        dataset: GenRecDataset,
        config: Optional[GenRecCollatorConfig] = None,
        seed: int = 42,
    ) -> None:
        """Configures the collator.

        Args:
            dataset (GenRecDataset): Dataset split from which examples are drawn. This is only
                used to initialize the negative sampler to get some global objects. You can also
                use this collator with other splits of the same dataset.
            config (Optional[GenRecCollatorConfig]): Optional collator configuration instance.
            seed (int): Random seed for the collator's internal RNG.
        """
        self._config = config or GenRecCollatorConfig()

        assert self._config.num_negative_samples >= 0, "num_negative_samples must be non-negative."
        self._negative_sampler = NegativeSamplerFactory.create(
            self._config.negative_sampling_strategy,
            dataset=dataset,
        )

        self._item_size = dataset.item_size
        self._sid_cache: Optional[Int[np.ndarray, "I+1 C"]] = dataset.sid_cache
        self._item_embeddings: Optional[Float[np.ndarray, "I+1 D"]] = dataset.item_embeddings
        if self._config.need_sid_tokens and self._sid_cache is None:  # pragma: no cover - defensive guard
            raise ValueError("Dataset must have SID cache when need_sid_tokens is True.")
        if self._config.need_embeddings and self._item_embeddings is None:  # pragma: no cover - defensive guard
            raise ValueError("Dataset must have item embeddings when need_embeddings is True.")

        need_pad_keys: Dict[str, type] = {
            "input_ids": np.int64,
            "timestamps": np.int64,
            "attention_mask": np.int64,
        }
        no_pad_keys: Dict[str, type] = {
            "user_id": np.int64,
            "labels": np.int64,
        }
        pad_values: Dict[str, np.generic] = {
            "input_ids": self._config.pad_item,
            "timestamps": np.int64(0),
            "attention_mask": np.int64(0),
        }
        if self._config.need_sid_tokens:
            need_pad_keys["input_sid_tokens"] = np.int64
            no_pad_keys["target_sid_tokens"] = np.int64
            pad_values["input_sid_tokens"] = self._config.pad_sid
        if self._config.need_embeddings:
            need_pad_keys["input_embeddings"] = np.float32
            no_pad_keys["target_embedding"] = np.float32
            pad_values["input_embeddings"] = np.float32(0.0)

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
        if self._config.need_sid_tokens and self._sid_cache is not None:
            negative_sid_tokens: Int[np.ndarray, "B N C"] = self._sid_cache[negative_item_ids]
            no_pad_batch["negative_sid_tokens"] = negative_sid_tokens
        if self._config.need_embeddings and self._item_embeddings is not None:
            negative_embeddings: Float[np.ndarray, "B N D"] = self._item_embeddings[negative_item_ids]
            no_pad_batch["negative_embeddings"] = negative_embeddings
