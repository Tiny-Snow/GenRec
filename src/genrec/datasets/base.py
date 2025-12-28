"""Base dataset for recommendation tasks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Generic, List, Literal, Mapping, Optional, Sequence, Tuple, Type, TypeVar, Union

import numpy as np
import pandas as pd
import torch
from jaxtyping import Float, Int
from torch.utils.data import Dataset

from .modules.lm_encoders import LMEncoder
from .modules.utils import SeedWorkerMixin, numpy_to_torch, pad_batch, stack_batch

__all__ = [
    "DatasetSplitLiteral",
    "RecCollator",
    "RecCollatorFactory",
    "RecCollatorConfig",
    "RecCollatorConfigFactory",
    "RecDataset",
    "RecDatasetFactory",
    "RecExample",
    "RecExampleFactory",
]


class DatasetSplitLiteral(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


_RecCollator = TypeVar("_RecCollator", bound="RecCollator[Any]")
_RecCollatorConfig = TypeVar("_RecCollatorConfig", bound="RecCollatorConfig")
_RecDataset = TypeVar("_RecDataset", bound="RecDataset[Any]")
_RecExample = TypeVar("_RecExample", bound="RecExample")


class RecExampleFactory:  # pragma: no cover - factory class
    """Factory for creating `RecExample` instances."""

    _registry: dict[str, Type[RecExample]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Type[_RecExample]], Type[_RecExample]]:
        """Decorator to register a `RecExample` implementation."""

        def decorator(example_cls: Type[_RecExample]) -> Type[_RecExample]:
            if name in cls._registry:
                raise ValueError(f"Dataset example '{name}' is already registered.")
            cls._registry[name] = example_cls
            return example_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs) -> RecExample:
        """Creates an instance of a registered `RecExample`."""
        if name not in cls._registry:
            raise ValueError(f"Dataset example '{name}' is not registered.")
        example_cls = cls._registry[name]
        return example_cls(**kwargs)


class RecDatasetFactory:  # pragma: no cover - factory class
    """Factory for creating `RecDataset` instances."""

    _registry: dict[str, Type[RecDataset[Any]]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Type[_RecDataset]], Type[_RecDataset]]:
        """Decorator to register a `RecDataset` implementation."""

        def decorator(dataset_cls: Type[_RecDataset]) -> Type[_RecDataset]:
            if name in cls._registry:
                raise ValueError(f"Dataset '{name}' is already registered.")
            cls._registry[name] = dataset_cls
            return dataset_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs) -> RecDataset[Any]:
        """Creates an instance of a registered `RecDataset`."""
        if name not in cls._registry:
            raise ValueError(f"Dataset '{name}' is not registered.")
        dataset_cls = cls._registry[name]
        return dataset_cls(**kwargs)


class RecCollatorConfigFactory:  # pragma: no cover - factory class
    """Factory for creating `RecCollatorConfig` instances."""

    _registry: dict[str, Type[RecCollatorConfig]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Type[_RecCollatorConfig]], Type[_RecCollatorConfig]]:
        """Decorator to register a `RecCollatorConfig` implementation."""

        def decorator(
            collator_config_cls: Type[_RecCollatorConfig],
        ) -> Type[_RecCollatorConfig]:
            if name in cls._registry:
                raise ValueError(f"Collator config '{name}' is already registered.")
            cls._registry[name] = collator_config_cls
            return collator_config_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs) -> RecCollatorConfig:
        """Creates an instance of a registered `RecCollatorConfig`."""
        if name not in cls._registry:
            raise ValueError(f"Collator config '{name}' is not registered.")
        collator_config_cls = cls._registry[name]
        return collator_config_cls(**kwargs)


class RecCollatorFactory:  # pragma: no cover - factory class
    """Factory for creating `RecCollator` instances."""

    _registry: dict[str, Type[RecCollator[Any]]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Type[_RecCollator]], Type[_RecCollator]]:
        """Decorator to register a `RecCollator` implementation."""

        def decorator(collator_cls: Type[_RecCollator]) -> Type[_RecCollator]:
            if name in cls._registry:
                raise ValueError(f"Collator '{name}' is already registered.")
            cls._registry[name] = collator_cls
            return collator_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs) -> RecCollator[Any]:
        """Creates an instance of a registered `RecCollator`."""
        if name not in cls._registry:
            raise ValueError(f"Collator '{name}' is not registered.")
        collator_cls = cls._registry[name]
        return collator_cls(**kwargs)


@dataclass(slots=True)
class RecExample:
    """Base class for recommendation examples."""

    pass


class RecDataset(Dataset[_RecExample], Generic[_RecExample], ABC):
    """Base dataset providing shared utilities for recommendation datasets.

    This class provides various common utilities for constructing recommendation datasets.
    Subclasses must implement the `_build_examples` method to materialise dataset examples.
    """

    def __init__(
        self,
        interaction_data_path: Union[pd.DataFrame, str, Path],
        split: DatasetSplitLiteral,
        max_seq_length: int = 20,
        min_seq_length: int = 1,
        sid_cache: Optional[Int[np.ndarray, "I+1 C"]] = None,
        textual_data_path: Optional[Union[pd.DataFrame, str, Path]] = None,
        lm_encoder: Optional[LMEncoder] = None,
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
        """
        if split not in {
            "train",
            "validation",
            "test",
        }:  # pragma: no cover - defensive guard
            raise ValueError(f"Unsupported split: {split}")

        self._split = split
        self._max_seq_length = int(max_seq_length)
        self._min_seq_length = int(min_seq_length)
        self._sid_cache = sid_cache

        self._item_embeddings: Optional[Float[np.ndarray, "I+1 D"]] = None
        if textual_data_path is not None:
            if lm_encoder is None:  # pragma: no cover - defensive guard
                raise ValueError("textual_data_path provided without lm_encoder.")
            assert isinstance(lm_encoder, LMEncoder)
            self._item_embeddings = self._build_item_embeddings(textual_data_path, lm_encoder)

        (
            self._user_interactions,
            self._user_interaction_timestamps,
            self._user_positive_items,
        ) = self._build_interactions(interaction_data_path)

        self._examples = self._build_examples()

        self._item_popularity = self._compute_item_popularity()

    @staticmethod
    def _load_dataframe(
        data_source: Union[pd.DataFrame, str, Path],
        columns: Sequence[str],
        dtypes: Mapping[str, Any],
    ) -> pd.DataFrame:
        """Loads a dataframe from memory or disk and ensures required columns and dtypes.

        Args:
            data_source (Union[pd.DataFrame, str, Path]): Pandas DataFrame or path to a pickle file
                containing `UserID`, `ItemID`, and `Timestamp` (Unix time).
            columns (Sequence[str]): Required column names.
            dtypes (Mapping[str, Any]): Expected dtypes per column.
        """
        if isinstance(data_source, pd.DataFrame):
            frame = data_source.copy(deep=False)
        else:  # pragma: no cover - load from file
            path = Path(data_source)
            if not path.exists():
                raise FileNotFoundError(f"Data file not found: {path}")
            if path.suffix != ".pkl":
                raise ValueError("Only Pandas pickle (.pkl) files are supported.")
            frame = pd.read_pickle(path)

        missing = set(columns).difference(frame.columns)
        if missing:  # pragma: no cover - missing required columns
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        for col, dtype in dtypes.items():
            if frame[col].dtype != dtype:  # pragma: no cover - convert dtype
                try:
                    frame[col] = frame[col].astype(dtype)
                except Exception as exc:
                    raise ValueError(f"Failed to convert column {col} to {dtype}.") from exc

        return frame

    def _build_item_embeddings(
        self,
        textual_data_path: Union[pd.DataFrame, str, Path],
        lm_encoder: LMEncoder,
    ) -> Float[np.ndarray, "I+1 D"]:
        """Encodes item titles into dense vectors using `encoder`.
        Returns a `np.ndarray` with shape (num_items, embedding_dim).
        """
        textual_frame = self._load_dataframe(
            textual_data_path,
            columns=["ItemID", "Title"],
            dtypes={"ItemID": np.int64, "Title": object},
        )
        num_items = int(textual_frame["ItemID"].max())
        assert textual_frame["ItemID"].min() == 1, "ItemIDs must start from 1."
        assert textual_frame["ItemID"].nunique() == num_items, "ItemIDs must be contiguous integers."

        titles = textual_frame["Title"].to_list()
        embeddings = lm_encoder.encode(titles).astype(np.float32, copy=False)
        padding_embedding = np.zeros((1, embeddings.shape[1]), dtype=np.float32)
        embeddings = np.vstack([padding_embedding, embeddings])

        return embeddings

    def _build_interactions(
        self,
        data_source: Union[pd.DataFrame, str, Path],
    ) -> Tuple[
        List[Int[np.ndarray, "_"]],
        List[Int[np.ndarray, "_"]],
        List[Int[np.ndarray, "_"]],
    ]:
        """Constructs per-user interactions and positive item sets.

        Args:
            data_source (Union[pd.DataFrame, str, Path]): Pandas DataFrame or path to a pickle file
                containing `UserID`, `ItemID`, and `Timestamp` (Unix time).

        Returns:
            Tuple[List[Int[np.ndarray, "_"]], List[Int[np.ndarray, "_"]], List[Int[np.ndarray, "_"]]]:
                Tuple of three lists describing per-user interactions:
                    `user_interactions`: List of numpy arrays indexed by user ID, each containing
                        the full interaction sequence sorted by timestamp chronologically.
                    `user_interaction_timestamps`: List of numpy arrays indexed by user ID, each
                        containing the timestamps of interactions sorted chronologically.
                    `user_positive_items`: List of numpy arrays indexed by user ID, each containing
                        the unique positive items sorted in ascending ID order.
        """
        data_frame = self._load_dataframe(
            data_source,
            columns=["UserID", "ItemID", "Timestamp"],
            dtypes={"UserID": np.int64, "ItemID": object, "Timestamp": object},
        )
        num_users = int(data_frame["UserID"].max()) + 1
        assert data_frame["UserID"].min() == 0, "UserIDs must start from 0."
        assert data_frame["UserID"].nunique() == num_users, "UserIDs must be contiguous integers."
        assert (
            data_frame["Timestamp"].apply(lambda ts: bool(np.all(np.asarray(ts)[:-1] <= np.asarray(ts)[1:]))).all()
        ), "Timestamps must be sorted in non-decreasing order per user."

        user_interactions = [np.empty(0, dtype=np.int64) for _ in range(num_users)]
        user_interaction_timestamps = [np.empty(0, dtype=np.int64) for _ in range(num_users)]
        user_positive_items = [np.empty(0, dtype=np.int64) for _ in range(num_users)]

        user_ids = data_frame["UserID"].to_numpy(dtype=np.int64)
        item_sequences = data_frame["ItemID"].to_list()
        timestamp_sequences = data_frame["Timestamp"].to_list()

        for user_id, sequence, timestamps in zip(user_ids, item_sequences, timestamp_sequences, strict=True):
            items = np.asarray(sequence, dtype=np.int64).reshape(-1)
            times = np.asarray(timestamps, dtype=np.int64).reshape(-1)
            assert items.shape == times.shape, "Item and timestamp sequences must have the same length."
            user_interactions[user_id] = items
            user_interaction_timestamps[user_id] = times
            user_positive_items[user_id] = np.unique(items)

        return user_interactions, user_interaction_timestamps, user_positive_items

    def _compute_item_popularity(self) -> Int[np.ndarray, "I+1"]:
        """Computes item popularity (non-normalised), i.e., the number of
        interactions per item, returning an array of shape (num_items + 1,).

        .. note::
            Note that the item popularity is calculated including all interactions,
            regardless of the dataset split.
        """
        popularity = np.zeros((self.item_size + 1,), dtype=np.int64)
        for items in self._user_interactions:
            for item in items:
                popularity[item] += 1
        return popularity

    def stats(self) -> Dict[str, Union[int, float]]:
        """Summarises sequence length statistics for the dataset."""
        num_examples = len(self._examples)
        num_users = self.user_size
        num_items = self.item_size

        interaction_lengths = [len(items) for items in self._user_interactions if len(items) > 0]
        avg_interaction_length = float(np.mean(interaction_lengths)) if interaction_lengths else 0.0
        max_interaction_length = int(np.max(interaction_lengths)) if interaction_lengths else 0
        min_interaction_length = int(np.min(interaction_lengths)) if interaction_lengths else 0

        return {
            "num_users": num_users,
            "num_items": num_items,
            "num_examples": num_examples,
            "avg_interaction_length": avg_interaction_length,
            "max_interaction_length": max_interaction_length,
            "min_interaction_length": min_interaction_length,
        }

    def __len__(self) -> int:
        """Returns the number of materialised examples."""
        return len(self._examples)

    def __getitem__(self, index: int) -> Any:
        """Retrieves the example at `index`."""
        return self._examples[index]

    @property
    def split(self) -> DatasetSplitLiteral:
        """Returns the dataset split."""
        return self._split

    @property
    def sid_cache(self) -> Optional[Int[np.ndarray, "I+1 C"]]:
        """Exposes the SID cache, when available."""
        return self._sid_cache

    @property
    def sid_width(self) -> Optional[int]:
        """Returns the width of SID sequences in the cache, when available."""
        if self._sid_cache is None:  # pragma: no cover - SID cache absent
            return None
        return self._sid_cache.shape[1]

    @property
    def item_embeddings(self) -> Optional[Float[np.ndarray, "I+1 D"]]:
        """Exposes the cached dense item embeddings, when available."""
        return self._item_embeddings

    @property
    def embedding_dim(self) -> Optional[int]:
        """Returns the dimensionality of cached embeddings, if present."""
        if self._item_embeddings is None:  # pragma: no cover - embedding absent
            return None
        return self._item_embeddings.shape[1]

    @property
    def item_size(self) -> int:
        """Returns the number of items, excluding padding item 0. If the whole item list
        is provided in `textual_data_path`, we infer the size from there; otherwise,
        we estimate it from the maximum item ID observed in the interaction data.
        """
        if self._item_embeddings is not None:
            return self._item_embeddings.shape[0] - 1

        user_max_item_ids = [items[-1] if items.size > 0 else 0 for items in self._user_positive_items]
        return int(max(user_max_item_ids))

    @property
    def user_size(self) -> int:
        """Returns the number of users in the dataset."""
        return len(self._user_interactions)

    @property
    def user_interactions(self) -> List[Int[np.ndarray, "_"]]:
        """Returns the list of user interaction sequences."""
        return self._user_interactions

    @property
    def user_interaction_timestamps(self) -> List[Int[np.ndarray, "_"]]:
        """Returns the list of user interaction timestamp sequences."""
        return self._user_interaction_timestamps

    @property
    def user_positive_items(self) -> List[Int[np.ndarray, "_"]]:
        """Returns the list of user positive item sets."""
        return self._user_positive_items

    @property
    def item_popularity(self) -> Int[np.ndarray, "I+1"]:
        """Returns the item popularity array."""
        return self._item_popularity

    @property
    def train_item_popularity(self) -> Int[np.ndarray, "I+1"]:
        """Returns the item popularity computed from the training set only.

        .. note::
            By default, this property returns the same as `item_popularity`.
            Subclasses may override this method to provide split-specific popularity.
        """
        return self._item_popularity

    @abstractmethod
    def _build_examples(
        self,
    ) -> List[_RecExample]:  # pragma: no cover - abstract method
        """Materialises dataset examples from the interaction data."""
        ...


@dataclass
class RecCollatorConfig:
    """Base class for collator configuration dataclasses.

    Attributes:
        pad_item: Padding value for item ID.
        pad_label: Padding value for label (however not used).
    """

    @property
    def pad_item(self) -> np.int64:  # pragma: no cover - default implementation
        return np.int64(0)

    @property
    def pad_label(self) -> np.int64:  # pragma: no cover - default implementation
        return np.int64(-100)


class RecCollator(Generic[_RecExample], SeedWorkerMixin):
    """Base class for recommendation data collators. You may extend this class
    to implement custom collation logic by overriding the pre- and post-processing
    hooks, i.e., `_process_before_padding` and `_process_after_padding`.
    """

    def __init__(
        self,
        need_pad_keys: Dict[str, type],
        no_pad_keys: Dict[str, type],
        pad_values: Dict[str, np.generic],
        seed: int = 42,
    ) -> None:
        """Configures the collator.

        Args:
            need_pad_keys (Dict[str, type]): Keys that require padding, mapped to their numpy dtypes.
            no_pad_keys (Dict[str, type]): Keys that do not require padding, mapped to their numpy dtypes.
            pad_values (Dict[str, np.generic]): Padding values per field, e.g., {"field1": 0,
                "field2": -100}. If a field is missing, defaults to 0.
            seed (int): Random seed for the collator's internal RNG.
        """
        SeedWorkerMixin.__init__(self, global_seed=seed)

        self._need_pad_keys = need_pad_keys
        self._no_pad_keys = no_pad_keys
        self._pad_values = pad_values

    def __call__(self, examples: List[_RecExample]) -> Dict[str, torch.Tensor]:
        """Pads and stacks `examples` into a training batch.
        Here we provide a generic implementation where two hooks `_process_before_padding`
        and `_process_after_padding` can be overridden by subclasses to implement
        custom processing logic before and after padding.

        Args:
            examples (List[_RecExample]): List of dataclass instances to collate.

        Returns:
            Dict[str, torch.Tensor]: Dictionary mapping field names to PyTorch tensors representing
                the padded and stacked batch.
        """
        batch_seed = self.next_batch_seed()
        batch = self._examples_to_numpy(examples)

        # Optional pre-processing before padding.
        self._process_before_padding(batch, batch_seed=batch_seed)

        need_pad_batch = [{key: sample[key] for key in self._need_pad_keys} for sample in batch]
        need_pad_batch = pad_batch(need_pad_batch, direction="left", pad_values=self._pad_values)
        no_pad_batch = [{key: sample[key] for key in sample if key in self._no_pad_keys} for sample in batch]
        no_pad_batch = stack_batch(no_pad_batch)

        # Optional post-processing after padding.
        self._process_after_padding(need_pad_batch, no_pad_batch, batch_seed=batch_seed)

        combined_batch = {**need_pad_batch, **no_pad_batch}
        return numpy_to_torch(combined_batch)

    def _process_before_padding(
        self, batch: List[Dict[str, np.ndarray]], batch_seed: int
    ) -> None:  # pragma: no cover - optional hook
        """Hook for optional pre-processing of the batch before padding.
        You may modify `batch` in-place, e.g., adding new fields.

        .. note::
            You should not update `self._need_pad_keys`, `self._no_pad_keys`,
            or `self._pad_values` in this method, as this may lead to inconsistent
            behavior during multi-worker data loading. Instead, set up all required
            keys beforehand in the constructor.

        Args:
            batch (List[Dict[str, np.ndarray]]): List of samples, each a dict like
                {"field1": [...], "field2": [...]}.
            batch_seed (int): Random seed for the current batch.
        """
        pass

    def _process_after_padding(
        self,
        need_pad_batch: Dict[str, np.ndarray],
        no_pad_batch: Dict[str, np.ndarray],
        batch_seed: int,
    ) -> None:  # pragma: no cover - optional hook
        """Hook for optional post-processing of the batch after padding.
        You may modify `need_pad_batch` and `no_pad_batch` in-place,
        e.g., adding new fields.

        Args:
            need_pad_batch (Dict[str, np.ndarray]): Batch data for fields that require padding.
            no_pad_batch (Dict[str, np.ndarray]): Batch data for fields that do not require padding.
            batch_seed (int): Random seed for the current batch.
        """
        pass

    def _examples_to_numpy(self, examples: List[_RecExample]) -> List[Dict[str, np.ndarray]]:
        """Converts list of `DataclassInstance` examples to list of numpy dicts, excluding
        those keys that are not specified in `self._need_pad_keys` or `self._no_pad_keys`

        .. note::
            In this function, we do not check for the presence of all required keys in each
            example, which may be added in future.
        """
        batch = [asdict(example) for example in examples]
        all_keys = {**self._need_pad_keys, **self._no_pad_keys}
        for sample in batch:
            for key in list(sample.keys()):
                if key not in all_keys:  # pragma: no cover - remove unused keys
                    sample.pop(key)
            for key in all_keys:
                if key in sample:
                    sample[key] = np.asarray(sample[key], dtype=all_keys[key])
        return batch
