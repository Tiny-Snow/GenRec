from __future__ import annotations

import numpy as np
import pytest
import torch

from genrec.datasets.modules.negative_samplers import PopularityNegativeSampler, UniformNegativeSampler
from genrec.datasets.modules.utils import SeedWorkerMixin, numpy_to_torch, pad_batch, stack_batch


def test_pad_batch_supports_left_right_and_multiples():
    batch = [
        {
            "ids": np.array([1, 2], dtype=np.int64),
            "mask": np.array([1, 1], dtype=np.int32),
        },
        {"ids": np.array([3], dtype=np.int64), "mask": np.array([1], dtype=np.int32)},
    ]

    padded_left = pad_batch(batch, direction="left", pad_values={"ids": -1, "mask": 0})
    assert padded_left["ids"].tolist() == [[1, 2], [-1, 3]]
    assert padded_left["mask"].tolist() == [[1, 1], [0, 1]]

    padded_right = pad_batch(batch, direction="right", pad_values={"ids": 0})
    assert padded_right["ids"].tolist() == [[1, 2], [3, 0]]

    long_batch = [
        {"ids": np.array([1, 2, 3], dtype=np.int64)},
        {"ids": np.array([4, 5], dtype=np.int64)},
    ]
    padded_multiple = pad_batch(long_batch, pad_to_multiple_of=4, pad_values={"ids": 0})
    assert padded_multiple["ids"].shape == (2, 4)

    with pytest.raises(AssertionError):
        pad_batch(batch, direction="middle")


def test_pad_batch_respects_max_length():
    batch = [
        {"ids": np.array([1, 2], dtype=np.int64)},
        {"ids": np.array([3], dtype=np.int64)},
    ]
    padded = pad_batch(batch, direction="right", pad_values={"ids": 0}, max_length=5)
    assert padded["ids"].shape == (2, 5)
    assert padded["ids"].tolist() == [[1, 2, 0, 0, 0], [3, 0, 0, 0, 0]]


def test_stack_batch_and_numpy_to_torch():
    batch = [
        {"values": np.array([1, 2], dtype=np.int64)},
        {"values": np.array([3, 4], dtype=np.int64)},
    ]
    stacked = stack_batch(batch)
    assert stacked["values"].shape == (2, 2)
    assert np.array_equal(stacked["values"], np.array([[1, 2], [3, 4]], dtype=np.int64))

    torch_batch = numpy_to_torch(stacked)
    assert isinstance(torch_batch["values"], torch.Tensor)
    assert torch_batch["values"].dtype == torch.int64


def test_seed_worker_mixin_reproducibility():
    class DummyWorker(SeedWorkerMixin):
        def __init__(self, seed: int) -> None:
            super().__init__(global_seed=seed)

    worker_a = DummyWorker(123)
    worker_b = DummyWorker(123)
    worker_c = DummyWorker(321)

    seq_a = [worker_a.next_batch_seed() for _ in range(3)]
    seq_b = [worker_b.next_batch_seed() for _ in range(3)]
    seq_c = [worker_c.next_batch_seed() for _ in range(3)]

    assert seq_a == seq_b
    assert seq_c != seq_a


def test_uniform_negative_sampler_behaviour():
    class DummyDataset:
        def __init__(self, item_size: int) -> None:
            self.item_size = item_size

    sampler = UniformNegativeSampler(dataset=DummyDataset(item_size=20))
    history = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)

    negatives = sampler(history, num_samples=4, batch_seed=42)
    assert negatives.shape == (2, 4)
    for row, hist in zip(negatives, history, strict=True):
        assert set(row.tolist()).isdisjoint(set(hist.tolist()))

    repeated = sampler(history, num_samples=4, batch_seed=42)
    assert np.array_equal(negatives, repeated)

    different = sampler(history, num_samples=4, batch_seed=43)
    assert not np.array_equal(negatives, different)


def test_popularity_negative_sampler_behaviour():
    class DummyDataset:
        def __init__(self, popularity: np.ndarray) -> None:
            self.item_popularity = popularity
            self.train_item_popularity = popularity
            self.item_size = popularity.shape[0] - 1

    popularity = np.array([0, 1, 5, 2, 0, 3], dtype=np.int64)
    sampler = PopularityNegativeSampler(dataset=DummyDataset(popularity=popularity))
    history = np.array([[1, 2, 3], [3, 5, 5]], dtype=np.int32)

    negatives = sampler(history, num_samples=6, batch_seed=7)
    assert negatives.shape == (2, 6)
    assert np.all((negatives >= 1) & (negatives <= popularity.shape[0] - 1))
    assert not np.any(negatives == 4)
    for row, hist in zip(negatives, history, strict=True):
        assert set(row.tolist()).isdisjoint(set(hist.tolist()))

    repeated = sampler(history, num_samples=6, batch_seed=7)
    assert np.array_equal(negatives, repeated)

    different = sampler(history, num_samples=6, batch_seed=8)
    assert not np.array_equal(negatives, different)


def test_pad_batch_handles_empty_samples():
    batch = [{} for _ in range(2)]
    padded = pad_batch(batch)
    assert padded == {}


def test_pad_batch_pad_to_multiple_of_skips_redundant_padding():
    batch = [
        {"ids": np.array([1, 2], dtype=np.int64)},
        {"ids": np.array([3, 4], dtype=np.int64)},
    ]

    padded = pad_batch(batch, pad_to_multiple_of=2, pad_values={"ids": 0})

    assert padded["ids"].shape == (2, 2)
    assert padded["ids"].tolist() == [[1, 2], [3, 4]]
