from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from genrec.datasets import (
    DatasetSplitLiteral,
    GenRecCollator,
    GenRecCollatorConfig,
    GenRecDataset,
    QuantizerCollator,
    QuantizerDataset,
    SeqRecCollator,
    SeqRecCollatorConfig,
    SeqRecDataset,
)


@dataclass
class DummyEncoder:
    embedding_dim: int = 4

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        base = np.arange(len(texts) * self.embedding_dim, dtype=np.float32)
        return base.reshape(len(texts), self.embedding_dim)


def _make_large_interaction_frame(num_users: int, seq_len: int = 4, item_pool: int | None = None) -> pd.DataFrame:
    if item_pool is None:
        item_pool = max(num_users, seq_len)
    user_ids = np.arange(num_users, dtype=np.int64)
    sequences = []
    timestamps = []
    for user_id in user_ids:
        start = (user_id * seq_len) % item_pool
        sequence = [((start + offset) % item_pool) + 1 for offset in range(seq_len)]
        sequences.append(sequence)
        time_sequence = [user_id * seq_len + offset for offset in range(seq_len)]
        timestamps.append(time_sequence)
    return pd.DataFrame({"UserID": user_ids, "ItemID": sequences, "Timestamp": timestamps})


def _make_textual_frame(item_pool: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ItemID": np.arange(1, item_pool + 1, dtype=np.int64),
            "Title": [f"Item {idx}" for idx in range(1, item_pool + 1)],
        }
    )


def _make_short_interaction_frame(length: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "UserID": np.array([0], dtype=np.int64),
            "ItemID": [np.arange(1, length + 1, dtype=np.int64).tolist()],
            "Timestamp": [np.arange(length, dtype=np.int64).tolist()],
        }
    )


def _expected_batch_sizes(dataset_len: int, batch_size: int) -> list[int]:
    full_batches, remainder = divmod(dataset_len, batch_size)
    sizes = [batch_size] * full_batches
    if remainder:
        sizes.append(remainder)
    return sizes


def _assert_genrec_batches(loader: DataLoader, expected_sizes: list[int], num_negatives: int) -> None:
    batches = list(loader)
    assert len(batches) == len(expected_sizes)
    for idx, batch in enumerate(batches):
        expected_size = expected_sizes[idx]
        assert batch["user_id"].shape == (expected_size,)
        assert batch["labels"].shape == (expected_size,)
        assert batch["input_ids"].shape[0] == expected_size
        assert batch["attention_mask"].shape == batch["input_ids"].shape
        assert batch["timestamps"].shape == batch["input_ids"].shape
        assert batch["input_ids"].shape[1] <= 4
        assert batch["negative_item_ids"].shape == (expected_size, num_negatives)
        assert batch["negative_item_ids"].dtype == torch.int32


def _assert_seqrec_batches(loader: DataLoader, expected_sizes: list[int], num_negatives: int) -> None:
    batches = list(loader)
    assert len(batches) == len(expected_sizes)
    for idx, batch in enumerate(batches):
        expected_size = expected_sizes[idx]
        assert batch["user_id"].shape == (expected_size,)
        assert batch["input_ids"].shape[0] == expected_size
        assert batch["labels"].shape == batch["input_ids"].shape
        assert batch["attention_mask"].shape == batch["input_ids"].shape
        assert batch["timestamps"].shape == batch["input_ids"].shape
        assert batch["negative_item_ids"].shape == (expected_size, num_negatives)
        assert batch["negative_item_ids"].dtype == torch.int32


def _assert_quantizer_batches(loader: DataLoader, expected_sizes: list[int], embedding_dim: int) -> None:
    batches = list(loader)
    assert len(batches) == len(expected_sizes)
    for idx, batch in enumerate(batches):
        expected_size = expected_sizes[idx]
        assert batch["item_id"].shape == (expected_size,)
        assert batch["item_embedding"].shape == (expected_size, embedding_dim)
        assert batch["item_embedding"].dtype == torch.float32


@pytest.fixture()
def interaction_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "UserID": np.array([0, 1, 2], dtype=np.int64),
            "ItemID": [
                [1, 2, 3, 4, 5],
                [2, 3, 4, 5, 6],
                [3, 4, 5, 6, 7],
            ],
            "Timestamp": [
                [100, 101, 102, 103, 104],
                [110, 111, 112, 113, 114],
                [120, 121, 122, 123, 124],
            ],
        }
    )


@pytest.fixture()
def textual_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ItemID": np.arange(1, 8, dtype=np.int64),
            "Title": [f"Item {idx}" for idx in range(1, 8)],
        }
    )


@pytest.fixture()
def sid_cache() -> np.ndarray:
    sid_width = 3
    cache = np.zeros((8, sid_width), dtype=np.int64)
    # Populate rows 1..7 with unique token patterns.
    for item_id in range(1, 8):
        cache[item_id] = np.array([item_id, item_id + 10, item_id + 20], dtype=np.int64)
    return cache


@pytest.fixture()
def dummy_encoder() -> DummyEncoder:
    return DummyEncoder()


@pytest.fixture()
def genrec_dataset(interaction_frame, textual_frame, sid_cache, dummy_encoder) -> GenRecDataset:
    return GenRecDataset(
        interaction_data_path=interaction_frame,
        split=DatasetSplitLiteral.TRAIN,
        max_seq_length=3,
        min_seq_length=1,
        sid_cache=sid_cache,
        textual_data_path=textual_frame,
        lm_encoder=dummy_encoder,
    )


@pytest.fixture()
def seqrec_dataset(interaction_frame, sid_cache) -> SeqRecDataset:
    return SeqRecDataset(
        interaction_data_path=interaction_frame,
        split=DatasetSplitLiteral.TRAIN,
        max_seq_length=4,
        min_seq_length=2,
        sid_cache=sid_cache,
    )


@pytest.fixture()
def quantizer_dataset(interaction_frame, textual_frame, dummy_encoder) -> QuantizerDataset:
    return QuantizerDataset(
        interaction_data_path=interaction_frame,
        split=DatasetSplitLiteral.TRAIN,
        max_seq_length=4,
        min_seq_length=1,
        textual_data_path=textual_frame,
        lm_encoder=dummy_encoder,
    )


def test_genrec_dataset_examples(genrec_dataset, sid_cache, dummy_encoder):
    assert len(genrec_dataset) == 6
    assert genrec_dataset.user_size == 3
    assert genrec_dataset.item_size == 7
    assert genrec_dataset.sid_width == sid_cache.shape[1]
    assert genrec_dataset.embedding_dim == dummy_encoder.embedding_dim

    example = genrec_dataset[0]
    assert example.user_id in {0, 1, 2}
    assert example.input_ids.ndim == 1
    assert example.timestamps.shape == example.input_ids.shape
    assert example.input_sid_tokens.shape[1] == sid_cache.shape[1]
    assert example.target_sid_tokens.shape == (sid_cache.shape[1],)
    assert example.input_embeddings.shape[1] == dummy_encoder.embedding_dim
    assert example.target_embedding.shape[0] == dummy_encoder.embedding_dim

    stats = genrec_dataset.stats()
    assert stats["num_users"] == 3
    assert stats["num_items"] == 7
    assert stats["num_examples"] == 6

    interactions = genrec_dataset.user_interactions
    positives = genrec_dataset.user_positive_items
    assert len(interactions) == genrec_dataset.user_size == 3
    assert len(positives) == genrec_dataset.user_size == 3
    assert np.array_equal(interactions[0], np.array([1, 2, 3, 4, 5], dtype=np.int64))
    assert np.array_equal(positives[1], np.array([2, 3, 4, 5, 6], dtype=np.int64))


def test_genrec_collator_with_dataloader(genrec_dataset, dummy_encoder):
    config = GenRecCollatorConfig(num_negative_samples=2, need_sid_tokens=True, need_embeddings=True)
    collator = GenRecCollator(genrec_dataset, config=config, seed=123)

    max_item_id = genrec_dataset.item_size

    def fake_negative_sampler(history, num_samples, batch_seed=None):
        return np.full((history.shape[0], num_samples), fill_value=max_item_id, dtype=np.int32)

    collator._negative_sampler = fake_negative_sampler  # type: ignore[assignment]

    loader = DataLoader(genrec_dataset, batch_size=2, collate_fn=collator)
    batch = next(iter(loader))

    assert batch["input_ids"].shape[0] == 2
    assert batch["attention_mask"].shape == batch["input_ids"].shape
    assert batch["timestamps"].shape == batch["input_ids"].shape
    assert batch["input_sid_tokens"].shape[2] == genrec_dataset.sid_width
    assert batch["target_sid_tokens"].shape == (2, genrec_dataset.sid_width)
    assert batch["input_embeddings"].shape[2] == dummy_encoder.embedding_dim
    assert batch["target_embedding"].shape == (2, dummy_encoder.embedding_dim)
    assert batch["negative_item_ids"].shape == (2, 2)
    assert batch["negative_sid_tokens"].shape == (2, 2, genrec_dataset.sid_width)
    assert batch["negative_embeddings"].shape == (2, 2, dummy_encoder.embedding_dim)

    for tensor in batch.values():
        assert isinstance(tensor, torch.Tensor)


def test_genrec_collator_without_sid_or_embedding_features(interaction_frame):
    dataset = GenRecDataset(
        interaction_data_path=interaction_frame,
        split=DatasetSplitLiteral.TRAIN,
        max_seq_length=3,
        min_seq_length=1,
    )
    config = GenRecCollatorConfig(num_negative_samples=1, need_sid_tokens=False, need_embeddings=False)
    collator = GenRecCollator(dataset, config=config, seed=11)

    def fake_negative_sampler(history, num_samples, batch_seed=None):
        return np.full((history.shape[0], num_samples), fill_value=dataset.item_size, dtype=np.int32)

    collator._negative_sampler = fake_negative_sampler  # type: ignore[assignment]

    loader = DataLoader(dataset, batch_size=2, collate_fn=collator)
    batch = next(iter(loader))

    assert batch["negative_item_ids"].shape == (2, 1)
    assert "negative_sid_tokens" not in batch
    assert "negative_embeddings" not in batch


def test_seqrec_dataset_and_collator(seqrec_dataset):
    assert len(seqrec_dataset) == 3

    example = seqrec_dataset[0]
    assert example.input_ids.shape == example.labels.shape
    assert example.input_ids.shape[0] >= 2
    assert example.timestamps.shape == example.input_ids.shape

    config = SeqRecCollatorConfig(num_negative_samples=1)
    collator = SeqRecCollator(seqrec_dataset, config=config, seed=321)

    def fake_negative_sampler(history, num_samples, batch_seed=None):
        return np.ones((history.shape[0], num_samples), dtype=np.int32)

    collator._negative_sampler = fake_negative_sampler  # type: ignore[assignment]

    loader = DataLoader(seqrec_dataset, batch_size=3, collate_fn=collator)
    batch = next(iter(loader))

    assert batch["input_ids"].shape == batch["labels"].shape
    assert batch["attention_mask"].shape == batch["input_ids"].shape
    assert batch["timestamps"].shape == batch["input_ids"].shape
    assert batch["negative_item_ids"].shape == (3, 1)


def test_seqrec_dataset_train_item_popularity_matches_recomputed(seqrec_dataset):
    recomputed = seqrec_dataset._compute_train_item_popularity()
    np.testing.assert_array_equal(seqrec_dataset.train_item_popularity, recomputed)


def test_genrec_dataset_train_item_popularity_matches_recomputed(genrec_dataset):
    recomputed = genrec_dataset._compute_train_item_popularity()
    np.testing.assert_array_equal(genrec_dataset.train_item_popularity, recomputed)


def test_quantizer_dataset_train_item_popularity_defaults_to_global(quantizer_dataset):
    np.testing.assert_array_equal(quantizer_dataset.train_item_popularity, quantizer_dataset.item_popularity)


@pytest.mark.parametrize(
    ("split", "expected_context", "expected_target"),
    [
        (DatasetSplitLiteral.VALIDATION, [1, 2, 3], 4),
        (DatasetSplitLiteral.TEST, [1, 2, 3, 4], 5),
    ],
)
def test_genrec_iter_split_handles_eval_and_test(interaction_frame, split, expected_context, expected_target):
    dataset = GenRecDataset(
        interaction_data_path=interaction_frame,
        split=split,
        max_seq_length=5,
        min_seq_length=1,
    )
    items = dataset.user_interactions[0]
    times = dataset.user_interaction_timestamps[0]

    splits = list(dataset._iter_split(items, times))
    assert len(splits) == 1
    context, target, context_times = splits[0]

    assert context.tolist() == expected_context
    assert target == expected_target
    assert context_times.tolist() == times[: len(expected_context)].tolist()


@pytest.mark.parametrize("dataset_fixture", ["genrec_dataset", "seqrec_dataset"])
def test_dataset_item_popularity_matches_item_size(request, dataset_fixture):
    dataset = request.getfixturevalue(dataset_fixture)
    popularity = dataset.item_popularity
    assert popularity.shape == (dataset.item_size + 1,)
    assert popularity.dtype == np.int64


def test_quantizer_dataset_and_collator(quantizer_dataset, dummy_encoder):
    assert len(quantizer_dataset) == quantizer_dataset.item_size
    example = quantizer_dataset[0]
    assert example.item_embedding.shape[0] == dummy_encoder.embedding_dim

    collator = QuantizerCollator(quantizer_dataset)
    loader = DataLoader(quantizer_dataset, batch_size=4, collate_fn=collator)
    batch = next(iter(loader))

    assert batch["item_id"].shape == (4,)
    assert batch["item_embedding"].shape == (4, dummy_encoder.embedding_dim)
    assert batch["item_embedding"].dtype == torch.float32


def test_genrec_iter_split_requires_minimum_length_for_validation():
    frame = _make_short_interaction_frame(length=2)
    dataset = GenRecDataset(
        interaction_data_path=frame,
        split=DatasetSplitLiteral.VALIDATION,
        max_seq_length=3,
        min_seq_length=1,
    )
    items = dataset.user_interactions[0]
    times = dataset.user_interaction_timestamps[0]
    assert list(dataset._iter_split(items, times)) == []


def test_genrec_iter_split_requires_minimum_length_for_test():
    frame = _make_short_interaction_frame(length=1)
    dataset = GenRecDataset(
        interaction_data_path=frame,
        split=DatasetSplitLiteral.TEST,
        max_seq_length=3,
        min_seq_length=1,
    )
    items = dataset.user_interactions[0]
    times = dataset.user_interaction_timestamps[0]
    assert list(dataset._iter_split(items, times)) == []


def test_large_scale_multiworker_uniform_negative_sampler():
    num_users = 10032
    batch_size = 4096
    num_negatives = 256
    seq_len = 4
    item_pool = num_users

    interactions = _make_large_interaction_frame(num_users, seq_len=seq_len, item_pool=item_pool)
    textual = _make_textual_frame(item_pool)
    expected_sizes = _expected_batch_sizes(num_users, batch_size)
    expected_batches = math.ceil(num_users / batch_size)
    assert len(expected_sizes) == expected_batches

    splits = (
        DatasetSplitLiteral.TRAIN,
        DatasetSplitLiteral.VALIDATION,
        DatasetSplitLiteral.TEST,
    )

    for split in splits:
        dataset = GenRecDataset(
            interaction_data_path=interactions,
            split=split,
            max_seq_length=3,
            min_seq_length=1,
        )
        assert len(dataset) == num_users
        assert dataset.split == split
        collator = GenRecCollator(
            dataset,
            config=GenRecCollatorConfig(
                num_negative_samples=num_negatives,
                need_sid_tokens=False,
                need_embeddings=False,
            ),
            seed=2025,
        )
        assert collator._negative_sampler.__class__.__name__ == "UniformNegativeSampler"
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collator,
            num_workers=2,
            shuffle=False,
            multiprocessing_context="fork",
        )
        _assert_genrec_batches(loader, expected_sizes, num_negatives)
        del loader

    for split in splits:
        dataset = SeqRecDataset(
            interaction_data_path=interactions,
            split=split,
            max_seq_length=4,
            min_seq_length=1,
        )
        assert len(dataset) == num_users
        assert dataset.split == split
        collator = SeqRecCollator(
            dataset,
            config=SeqRecCollatorConfig(num_negative_samples=num_negatives),
            seed=2025,
        )
        assert collator._negative_sampler.__class__.__name__ == "UniformNegativeSampler"
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collator,
            num_workers=2,
            shuffle=False,
            multiprocessing_context="fork",
        )
        _assert_seqrec_batches(loader, expected_sizes, num_negatives)
        del loader

    encoder = DummyEncoder()
    dataset = QuantizerDataset(
        interaction_data_path=interactions,
        split=DatasetSplitLiteral.TRAIN,
        max_seq_length=4,
        min_seq_length=1,
        textual_data_path=textual,
        lm_encoder=encoder,
    )
    assert len(dataset) == item_pool
    assert dataset.split == DatasetSplitLiteral.TRAIN
    collator = QuantizerCollator(dataset)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=2,
        shuffle=False,
        multiprocessing_context="fork",
    )
    _assert_quantizer_batches(loader, _expected_batch_sizes(len(dataset), batch_size), encoder.embedding_dim)
    del loader
