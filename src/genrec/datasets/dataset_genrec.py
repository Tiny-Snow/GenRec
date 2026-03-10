"""Dataset for generative recommendation tasks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

from jaxtyping import Float, Int
import numpy as np
import pandas as pd
import torch

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
from .modules.prefix_tree import PrefixTree

__all__ = [
    "GenRecCollator",
    "GenRecCollatorConfig",
    "GenRecDataset",
    "GenRecExample",
]


@RecExampleFactory.register("genrec")
@dataclass(slots=True)
class GenRecExample(RecExample):
    """Container storing a single training pair with item-level and SID views
    that suit encoder-decoder style generative recommendation models.

    The example keeps both the original item identifiers (for bookkeeping) and
    their flattened Semantic ID (SID) token counterparts that are directly
    consumed by generative models.

    Attributes:
        user_id: Identifier of the user to which the example belongs.
        input_ids: Flattened SID tokens derived from `input_item_ids`.
        labels: SID tokens representing `label_item_ids`, right-padded with EOS token.
        input_item_ids: Interaction history expressed as item IDs.
        label_item_ids: Positive item ID that should follow the history.
        timestamps: Timestamps aligned with `input_item_ids` in Unix time.
        input_embeddings: Optional dense embedding matrix aligned with `input_item_ids`.
        target_embedding: Optional dense embedding vector for `label_item_ids`.
    """

    user_id: int
    input_ids: Int[np.ndarray, "L*C"]
    labels: Int[np.ndarray, "C+1"]
    input_item_ids: Int[np.ndarray, "L"]
    label_item_ids: int
    timestamps: Int[np.ndarray, "L"]
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
        truncation_strategy: str = "tail",
        decoder_start_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        reserved_sids: int = 0,
    ) -> None:
        """Initialises the dataset and materialises user-level metadata.

        Args:
            interaction_data_path (Union[pd.DataFrame, str, Path]): Pandas DataFrame or path to a
                pickle file containing `UserID` and `ItemID` columns. We assume that the `UserID`
                begins from 0 and that `ItemID` begins from 1, both being contiguous integers. The
                `ItemID` of 0 is reserved for padding.
            split (DatasetSplitLiteral): Dataset split controlling example generation strategy.
            max_seq_length (int): Maximum length of interaction histories.
            min_seq_length (int): Minimum length of interaction histories.
            sid_cache (Optional[Int[np.ndarray, "I+1 C"]]): Optional mapping from item ID to SID
                sequence, stored as numpy arrays.
            textual_data_path (Optional[Union[pd.DataFrame, str, Path]]): Optional DataFrame or
                pickle file with `ItemID` and `Title` columns.
            lm_encoder (Optional[LMEncoder]): Optional encoder used to transform item titles into
                dense embeddings.
            truncation_strategy (str): Strategy for truncating interaction histories, supported
                options are `"tail"` and `"slide"`. `"tail"` will directly truncate the user history
                to `max_seq_length`, then construct training examples with length of history from
                `min_seq_length` to (up to) `max_seq_length`. `"slide"` will use a sliding window of
                size `max_seq_length` over the entire user history to construct training examples.
                Defaults to `"tail"`.
            decoder_start_token_id (Optional[int]): Beginning-of-sequence value for SID tokens during
                decoding. If `None`, defaults to 0.
            eos_token_id (Optional[int]): End-of-sequence value for SID tokens. If `None`, defaults to 1.
            reserved_sids (int): Number of reserved SID tokens (e.g., for special tokens like pad, bos, eos).
                The actual SID are shifted by this value. Defaults to 0.
        """
        assert truncation_strategy in {"tail", "slide"}, f"Unsupported truncation strategy: {truncation_strategy}."
        self.truncation_strategy = truncation_strategy

        self.decoder_start_token_id = decoder_start_token_id if decoder_start_token_id is not None else 0
        self.eos_token_id = eos_token_id if eos_token_id is not None else 1

        if sid_cache is None:  # pragma: no cover - defensive guard
            raise ValueError("GenRecDataset requires `sid_cache` to materialise SID tokens.")

        # Shift SIDs, 0 to revserved_sids - 1 are reserved for special tokens
        self.reserved_sids = reserved_sids
        sid_cache = sid_cache + reserved_sids

        # Construct prefix tree for constrained decoding, padding with decoder_start_token_id and eos_token_id
        self._prefix_tree = PrefixTree.from_mapping(
            {
                item_id: [decoder_start_token_id] + sid_cache[item_id].tolist() + [eos_token_id]
                for item_id in range(1, sid_cache.shape[0])
            }
        )

        # Cache item ID lookup from SID sequences, not padding with eos_token_id
        self._sid2item: Dict[Tuple[int, ...], int] = {
            tuple(sid_cache[item_id].tolist()): item_id for item_id in range(1, sid_cache.shape[0])
        }

        # Build base RecDataset
        super().__init__(
            interaction_data_path,
            split,
            max_seq_length,
            min_seq_length,
            sid_cache,
            textual_data_path,
            lm_encoder,
        )

        # Recompute training set item popularity
        self._train_item_popularity = self._compute_train_item_popularity()

    def _build_examples(self) -> List[GenRecExample]:
        """Generates training pairs for encoder-decoder style models using sliding
        windows over interaction histories.
        """
        examples: List[GenRecExample] = []
        user_ids = np.arange(self.user_size, dtype=np.int64)
        for user_id, items, timestamps in zip(user_ids, self.user_interactions, self.user_interaction_timestamps):
            items, timestamps = self._tail_truncate(items, timestamps)
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

    def _tail_truncate(
        self,
        items: Int[np.ndarray, "..."],
        times: Int[np.ndarray, "..."],
    ) -> Tuple[Int[np.ndarray, "..."], Int[np.ndarray, "..."]]:
        """Truncates the interaction history by keeping the most recent interactions.
        We trim the history to `max_seq_length + 3` to account for the target and
        validation/test items.
        """
        if self.truncation_strategy == "tail":
            return items[-self._max_seq_length - 3 :], times[-self._max_seq_length - 3 :]
        else:
            return items, times

    def _construct_example(
        self,
        user_id: int,
        context: Int[np.ndarray, "..."],
        target: int,
        times: Int[np.ndarray, "..."],
    ) -> GenRecExample:
        """Constructs a GenRecExample from the provided context and target."""
        assert self._sid_cache is not None, "SID cache must be available after __init__ guard."

        input_item_ids = context.astype(np.int64, copy=True)
        label_item_id = int(target)
        timestamps = times.astype(np.int64, copy=True)

        sid_context: Int[np.ndarray, "L C"] = self._sid_cache[input_item_ids]
        sid_target: Int[np.ndarray, "C"] = self._sid_cache[label_item_id]
        flattened_context = sid_context.reshape(-1).astype(np.int64, copy=True)
        sid_target = sid_target.astype(np.int64, copy=True)

        # pad target with eos token
        pad_eos = np.full((1,), self.eos_token_id, dtype=np.int64)
        sid_target = np.concatenate([sid_target, pad_eos], axis=0)

        example = GenRecExample(
            user_id=user_id,
            input_ids=flattened_context,
            labels=sid_target,
            input_item_ids=input_item_ids,
            label_item_ids=label_item_id,
            timestamps=timestamps,
        )

        if self._item_textual_embeddings is not None:
            example.input_embeddings = self._item_textual_embeddings[input_item_ids]
            example.target_embedding = self._item_textual_embeddings[label_item_id]

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

    @property
    def sid2item(self) -> Dict[Tuple[int, ...], int]:
        """Returns the mapping from SID sequences to item IDs."""
        return self._sid2item

    def get_prefix_allowed_tokens_fn(
        self,
    ) -> Optional[Callable[[int, Int[torch.Tensor, "seq_len"]], List[int]]]:
        """Returns a function that can be used during constrained beam search to restrict the next token
        choices based on the prefix tree.

        Returns:
            Optional[Callable[[int, Int[torch.Tensor, "seq_len"]], List[int]]]: A function that takes a
                batch index and the current input IDs tensor, and returns a list of allowed next tokens.
                If no prefix tree is available, returns `None`.
        """

        def prefix_allowed_tokens(
            batch_id: int,
            input_ids: Int[torch.Tensor, "seq_len"],
        ) -> List[int]:
            prefix = input_ids.tolist()
            return self._prefix_tree.next_tokens(prefix)

        return prefix_allowed_tokens


@RecCollatorConfigFactory.register("genrec")
@dataclass
class GenRecCollatorConfig(RecCollatorConfig):
    """Runtime settings consumed by `GenRecCollator`.

    Attributes:
        pad_sid: Padding value for Semantic ID token.
        need_embeddings: Whether to collate dense embeddings if present. Note that
            if the dataset examples do not contain embeddings, this flag should
            be set to `False` to avoid errors.
    """

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
            input_ids: `Int[torch.Tensor, "B L*C"]`.
                Flattened SID tokens aligned with `input_item_ids`.
            labels: `Int[torch.Tensor, "B C+1"]`.
                SID tokens describing `label_item_ids`, right-padded with EOS token.
            input_item_ids: `Int[torch.Tensor, "B L"]`.
                Input item ID sequences used for bookkeeping/metrics.
            label_item_ids: `Int[torch.Tensor, "B"]`.
                Target item IDs.
            timestamps: `Int[torch.Tensor, "B L"]`.
                Input timestamp sequences.
            attention_mask: `Int[torch.Tensor, "B L*C"]`.
                Attention masks for the flattened SID tokens.
            input_embeddings: `Optional[Float[torch.Tensor, "B L D"]]`.
                Input dense embeddings, when present and needed.
            target_embedding: `Optional[Float[torch.Tensor, "B D"]]`.
                Target dense embeddings, when present and needed.
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

        self._item_size = dataset.item_size
        self._sid_width = dataset.sid_width
        self._item_embeddings: Optional[Float[np.ndarray, "I+1 D"]] = dataset.item_textual_embeddings
        if self._config.need_embeddings and self._item_embeddings is None:  # pragma: no cover - defensive guard
            raise ValueError("Dataset must have item embeddings when need_embeddings is True.")

        need_pad_keys: Dict[str, type] = {
            "input_ids": np.int64,
            "attention_mask": np.int64,
            "input_item_ids": np.int64,
            "timestamps": np.int64,
        }
        no_pad_keys: Dict[str, type] = {
            "user_id": np.int64,
            "label_item_ids": np.int64,
            "labels": np.int64,
        }
        pad_values: Dict[str, np.generic] = {
            "input_ids": self._config.pad_sid,
            "attention_mask": np.int64(0),
            "input_item_ids": self._config.pad_item,
            "timestamps": np.int64(0),
        }
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
        """Add attention masks aligned with flattened SID tokens before padding."""
        for sample in batch:
            seq_len = sample["input_ids"].shape[0]
            sample["attention_mask"] = np.ones((seq_len,), dtype=np.int64)
