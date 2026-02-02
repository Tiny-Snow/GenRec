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


def _make_aux_item_embeddings(item_pool: int, aux_dim: int) -> np.ndarray:
    values = np.arange((item_pool + 1) * aux_dim, dtype=np.float32)
    values = values.reshape(item_pool + 1, aux_dim)
    values[0] = 0.0
    return values


def _make_short_interaction_frame(length: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "UserID": np.array([0], dtype=np.int64),
            "ItemID": [np.arange(1, length + 1, dtype=np.int64).tolist()],
            "Timestamp": [np.arange(length, dtype=np.int64).tolist()],
        }
    )


def _make_sid_cache(item_pool: int, sid_width: int = 3) -> np.ndarray:
    cache = np.zeros((item_pool + 1, sid_width), dtype=np.int64)
    for item_id in range(1, item_pool + 1):
        cache[item_id] = np.arange(item_id, item_id + sid_width, dtype=np.int64)
    return cache


def _expected_batch_sizes(dataset_len: int, batch_size: int) -> list[int]:
    full_batches, remainder = divmod(dataset_len, batch_size)
    sizes = [batch_size] * full_batches
    if remainder:
        sizes.append(remainder)
    return sizes


def _assert_genrec_batches(loader: DataLoader, expected_sizes: list[int]) -> None:
    batches = list(loader)
    assert len(batches) == len(expected_sizes)
    for idx, batch in enumerate(batches):
        expected_size = expected_sizes[idx]
        assert batch["user_id"].shape == (expected_size,)
        assert batch["labels"].ndim == 2
        sid_width = batch["labels"].shape[1]
        assert batch["input_ids"].shape == batch["attention_mask"].shape
        assert batch["input_ids"].shape[0] == expected_size
        assert batch["input_item_ids"].shape == batch["timestamps"].shape
        assert batch["input_item_ids"].shape[0] == expected_size
        assert batch["input_ids"].shape[1] == batch["input_item_ids"].shape[1] * sid_width
        assert batch["labels"].shape == (expected_size, sid_width)


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


def _assert_quantizer_batches(
    loader: DataLoader,
    expected_sizes: list[int],
    embedding_dim: int,
    aux_embedding_dim: int | None = None,
) -> None:
    batches = list(loader)
    assert len(batches) == len(expected_sizes)
    for idx, batch in enumerate(batches):
        expected_size = expected_sizes[idx]
        assert batch["item_id"].shape == (expected_size,)
        assert batch["item_embedding"].shape == (expected_size, embedding_dim)
        assert batch["item_embedding"].dtype == torch.float32
        if aux_embedding_dim is not None:
            assert batch["aux_item_embedding"].shape == (expected_size, aux_embedding_dim)
            assert batch["aux_item_embedding"].dtype == torch.float32


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
    return _make_sid_cache(item_pool=7, sid_width=3)


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
    item_pool = int(textual_frame["ItemID"].max())
    aux_embeddings = _make_aux_item_embeddings(item_pool, aux_dim=3)
    return QuantizerDataset(
        interaction_data_path=interaction_frame,
        max_seq_length=4,
        min_seq_length=1,
        textual_data_path=textual_frame,
        lm_encoder=dummy_encoder,
        aux_item_embeddings=aux_embeddings,
    )


def test_quantizer_dataset_aux_properties(interaction_frame, textual_frame, dummy_encoder) -> None:
    item_pool = int(textual_frame["ItemID"].max())
    aux_embeddings = _make_aux_item_embeddings(item_pool, aux_dim=2)
    dataset = QuantizerDataset(
        interaction_data_path=interaction_frame,
        max_seq_length=4,
        min_seq_length=1,
        textual_data_path=textual_frame,
        lm_encoder=dummy_encoder,
        aux_item_embeddings=aux_embeddings,
    )

    assert dataset.aux_item_embeddings is not None
    np.testing.assert_allclose(dataset.aux_item_embeddings[1:], aux_embeddings[1:])
    assert dataset.aux_embedding_dim == aux_embeddings.shape[1]

    dataset_no_aux = QuantizerDataset(
        interaction_data_path=interaction_frame,
        max_seq_length=2,
        min_seq_length=1,
        textual_data_path=textual_frame,
        lm_encoder=dummy_encoder,
    )
    assert dataset_no_aux.aux_item_embeddings is None
    assert dataset_no_aux.aux_embedding_dim is None


def test_genrec_dataset_examples(genrec_dataset, sid_cache, dummy_encoder):
    assert len(genrec_dataset) == 6
    assert genrec_dataset.user_size == 3
    assert genrec_dataset.item_size == 7
    assert genrec_dataset.sid_width == sid_cache.shape[1]
    assert genrec_dataset.sid_cache.shape == sid_cache.shape
    assert genrec_dataset.textual_embedding_dim == dummy_encoder.embedding_dim

    example = genrec_dataset[0]
    assert example.user_id in {0, 1, 2}
    assert example.input_item_ids.ndim == 1
    assert example.timestamps.shape == example.input_item_ids.shape
    assert example.input_ids.shape == (example.input_item_ids.shape[0] * sid_cache.shape[1],)
    assert example.labels.shape == (sid_cache.shape[1],)
    assert example.label_item_ids in genrec_dataset.user_positive_items[example.user_id]
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
    config = GenRecCollatorConfig(need_embeddings=True)
    collator = GenRecCollator(genrec_dataset, config=config, seed=123)

    loader = DataLoader(genrec_dataset, batch_size=2, collate_fn=collator)
    batch = next(iter(loader))

    assert batch["input_ids"].shape[0] == 2
    assert batch["attention_mask"].shape == batch["input_ids"].shape
    assert batch["input_item_ids"].shape == batch["timestamps"].shape
    assert batch["input_ids"].shape[1] == batch["input_item_ids"].shape[1] * genrec_dataset.sid_width
    assert batch["labels"].shape == (2, genrec_dataset.sid_width)
    assert batch["label_item_ids"].shape == (2,)
    assert batch["input_embeddings"].shape[2] == dummy_encoder.embedding_dim
    assert batch["target_embedding"].shape == (2, dummy_encoder.embedding_dim)

    for tensor in batch.values():
        assert isinstance(tensor, torch.Tensor)


def test_genrec_collator_without_sid_or_embedding_features(interaction_frame, sid_cache):
    dataset = GenRecDataset(
        interaction_data_path=interaction_frame,
        split=DatasetSplitLiteral.TRAIN,
        max_seq_length=3,
        min_seq_length=1,
        sid_cache=sid_cache,
    )
    config = GenRecCollatorConfig(need_embeddings=False)
    collator = GenRecCollator(dataset, config=config, seed=11)

    loader = DataLoader(dataset, batch_size=2, collate_fn=collator)
    batch = next(iter(loader))

    assert "input_embeddings" not in batch
    assert "target_embedding" not in batch
    assert batch["labels"].shape[1] == sid_cache.shape[1]


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


def test_genrec_dataset_sid2item_mapping(genrec_dataset, sid_cache):
    mapping = genrec_dataset.sid2item
    assert len(mapping) == genrec_dataset.item_size

    expected = {tuple(sid_cache[item_id].tolist()): item_id for item_id in range(1, genrec_dataset.item_size + 1)}
    assert mapping == expected

    arbitrary_sid = tuple(sid_cache[3].tolist())
    assert mapping[arbitrary_sid] == 3


def test_genrec_dataset_prefix_allowed_tokens_fn(genrec_dataset, sid_cache):
    prefix_fn = genrec_dataset.get_prefix_allowed_tokens_fn()
    assert prefix_fn is not None

    allowed_from_root = prefix_fn(0, torch.tensor([], dtype=torch.int64))
    assert allowed_from_root == [genrec_dataset.pad_token_id]

    allowed_after_pad = prefix_fn(0, torch.tensor([genrec_dataset.pad_token_id], dtype=torch.int64))
    expected_after_pad = sorted(int(token) for token in sid_cache[1 : genrec_dataset.item_size + 1, 0])
    assert sorted(allowed_after_pad) == expected_after_pad

    test_item_id = 3
    sid_tokens = sid_cache[test_item_id]
    allowed_next = prefix_fn(0, torch.tensor([genrec_dataset.pad_token_id, int(sid_tokens[0])], dtype=torch.int64))
    assert allowed_next == [int(sid_tokens[1])]

    allowed_after_full_item = prefix_fn(
        0, torch.tensor([genrec_dataset.pad_token_id] + sid_tokens.tolist(), dtype=torch.int64)
    )
    assert allowed_after_full_item == []

    invalid_allowed = prefix_fn(0, torch.tensor([9999], dtype=torch.int64))
    assert invalid_allowed == []


def test_quantizer_dataset_train_item_popularity_defaults_to_global(quantizer_dataset):
    np.testing.assert_array_equal(quantizer_dataset.train_item_popularity, quantizer_dataset.item_popularity)


@pytest.mark.parametrize(
    ("split", "expected_context", "expected_target"),
    [
        (DatasetSplitLiteral.VALIDATION, [1, 2, 3], 4),
        (DatasetSplitLiteral.TEST, [1, 2, 3, 4], 5),
    ],
)
def test_genrec_iter_split_handles_eval_and_test(
    interaction_frame, sid_cache, split, expected_context, expected_target
):
    dataset = GenRecDataset(
        interaction_data_path=interaction_frame,
        split=split,
        max_seq_length=5,
        min_seq_length=1,
        sid_cache=sid_cache,
    )
    items = dataset.user_interactions[0]
    times = dataset.user_interaction_timestamps[0]

    splits = list(dataset._iter_split(items, times))
    assert len(splits) == 1
    context, target, context_times = splits[0]

    assert context.tolist() == expected_context
    assert target == expected_target
    assert context_times.tolist() == times[: len(expected_context)].tolist()


def test_genrec_item_size_prefers_textual_titles_without_encoding(dummy_encoder):
    interaction_frame = _make_short_interaction_frame(length=3)
    textual_frame = _make_textual_frame(item_pool=6)
    sid_cache_short = _make_sid_cache(item_pool=3)
    sid_cache_text = _make_sid_cache(item_pool=6)

    dataset_text_only = GenRecDataset(
        interaction_data_path=interaction_frame,
        split=DatasetSplitLiteral.TRAIN,
        max_seq_length=4,
        min_seq_length=1,
        textual_data_path=textual_frame,
        sid_cache=sid_cache_text,
    )
    assert dataset_text_only.item_size == 6
    assert dataset_text_only.item_textual_embeddings is None

    dataset_no_textual = GenRecDataset(
        interaction_data_path=interaction_frame,
        split=DatasetSplitLiteral.TRAIN,
        max_seq_length=4,
        min_seq_length=1,
        sid_cache=sid_cache_short,
    )
    assert dataset_no_textual.item_size == 3

    dataset_with_encoder = GenRecDataset(
        interaction_data_path=interaction_frame,
        split=DatasetSplitLiteral.TRAIN,
        max_seq_length=4,
        min_seq_length=1,
        textual_data_path=textual_frame,
        lm_encoder=dummy_encoder,
        sid_cache=sid_cache_text,
    )
    assert dataset_with_encoder.item_size == 6
    assert dataset_with_encoder.item_textual_embeddings is not None
    assert dataset_with_encoder.textual_embedding_dim == dummy_encoder.embedding_dim


def test_genrec_tail_truncation_keeps_recent_history_only():
    frame = _make_short_interaction_frame(length=12)
    raw_items = np.array(frame.iloc[0]["ItemID"], dtype=np.int64)
    raw_times = np.array(frame.iloc[0]["Timestamp"], dtype=np.int64)
    max_seq_length = 4
    sid_cache = _make_sid_cache(item_pool=int(raw_items.max()))

    dataset = GenRecDataset(
        interaction_data_path=frame,
        split=DatasetSplitLiteral.TRAIN,
        max_seq_length=max_seq_length,
        min_seq_length=1,
        truncation_strategy="tail",
        sid_cache=sid_cache,
    )

    truncated_items, truncated_times = dataset._tail_truncate(raw_items, raw_times)
    np.testing.assert_array_equal(truncated_items, raw_items[-(max_seq_length + 3) :])
    np.testing.assert_array_equal(truncated_times, raw_times[-(max_seq_length + 3) :])
    assert len(dataset) == max_seq_length

    concatenated_contexts = np.concatenate([example.input_item_ids for example in dataset])
    assert concatenated_contexts.min() == truncated_items[0]


def test_genrec_slide_truncation_retains_full_history_for_windows():
    frame = _make_short_interaction_frame(length=12)
    raw_items = np.array(frame.iloc[0]["ItemID"], dtype=np.int64)
    raw_times = np.array(frame.iloc[0]["Timestamp"], dtype=np.int64)
    max_seq_length = 4
    sid_cache = _make_sid_cache(item_pool=int(raw_items.max()))

    dataset = GenRecDataset(
        interaction_data_path=frame,
        split=DatasetSplitLiteral.TRAIN,
        max_seq_length=max_seq_length,
        min_seq_length=1,
        truncation_strategy="slide",
        sid_cache=sid_cache,
    )

    truncated_items, truncated_times = dataset._tail_truncate(raw_items, raw_times)
    np.testing.assert_array_equal(truncated_items, raw_items)
    np.testing.assert_array_equal(truncated_times, raw_times)
    assert len(dataset) == raw_items.shape[0] - 3

    concatenated_contexts = np.concatenate([example.input_item_ids for example in dataset])
    assert concatenated_contexts.min() == raw_items[0]
    assert dataset[0].input_item_ids.tolist() == [int(raw_items[0])]


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
    assert example.aux_item_embedding is not None
    assert example.aux_item_embedding.ndim == 1
    aux_dim = example.aux_item_embedding.shape[0]

    collator = QuantizerCollator(quantizer_dataset)
    loader = DataLoader(quantizer_dataset, batch_size=4, collate_fn=collator)
    batch = next(iter(loader))

    assert batch["item_id"].shape == (4,)
    assert batch["item_embedding"].shape == (4, dummy_encoder.embedding_dim)
    assert batch["item_embedding"].dtype == torch.float32
    assert batch["aux_item_embedding"].shape == (4, aux_dim)
    assert batch["aux_item_embedding"].dtype == torch.float32


def test_genrec_iter_split_requires_minimum_length_for_validation():
    frame = _make_short_interaction_frame(length=2)
    sid_cache = _make_sid_cache(item_pool=2)
    dataset = GenRecDataset(
        interaction_data_path=frame,
        split=DatasetSplitLiteral.VALIDATION,
        max_seq_length=3,
        min_seq_length=1,
        sid_cache=sid_cache,
    )
    items = dataset.user_interactions[0]
    times = dataset.user_interaction_timestamps[0]
    assert list(dataset._iter_split(items, times)) == []


def test_genrec_iter_split_requires_minimum_length_for_test():
    frame = _make_short_interaction_frame(length=1)
    sid_cache = _make_sid_cache(item_pool=1)
    dataset = GenRecDataset(
        interaction_data_path=frame,
        split=DatasetSplitLiteral.TEST,
        max_seq_length=3,
        min_seq_length=1,
        sid_cache=sid_cache,
    )
    items = dataset.user_interactions[0]
    times = dataset.user_interaction_timestamps[0]
    assert list(dataset._iter_split(items, times)) == []


def test_large_scale_multiworker_uniform_negative_sampler():
    num_users = 10032
    batch_size = 4096
    seq_len = 4
    item_pool = num_users
    sid_width = 2

    interactions = _make_large_interaction_frame(num_users, seq_len=seq_len, item_pool=item_pool)
    textual = _make_textual_frame(item_pool)
    sid_cache = _make_sid_cache(item_pool=item_pool, sid_width=sid_width)
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
            sid_cache=sid_cache,
        )
        assert len(dataset) == num_users
        assert dataset.split == split
        collator = GenRecCollator(dataset, config=GenRecCollatorConfig(need_embeddings=False), seed=2025)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collator,
            num_workers=2,
            shuffle=False,
            multiprocessing_context="fork",
        )
        _assert_genrec_batches(loader, expected_sizes)
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
            config=SeqRecCollatorConfig(num_negative_samples=256),
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
        _assert_seqrec_batches(loader, expected_sizes, 256)
        del loader

    encoder = DummyEncoder()
    aux_embeddings = _make_aux_item_embeddings(item_pool, aux_dim=5)
    dataset = QuantizerDataset(
        interaction_data_path=interactions,
        max_seq_length=4,
        min_seq_length=1,
        textual_data_path=textual,
        lm_encoder=encoder,
        aux_item_embeddings=aux_embeddings,
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
    _assert_quantizer_batches(
        loader,
        _expected_batch_sizes(len(dataset), batch_size),
        encoder.embedding_dim,
        aux_embedding_dim=aux_embeddings.shape[1],
    )
    del loader
