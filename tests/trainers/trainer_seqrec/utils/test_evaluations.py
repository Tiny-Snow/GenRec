from typing import Sequence

import numpy as np
import pytest
import torch

from genrec.trainers.trainer_seqrec.utils.evaluations import (
    SeqRecMetricFactory,
    calc_metric_arp,
    calc_metric_gini,
    calc_metric_hr,
    calc_metric_ndcg,
    calc_metric_popularity,
    calc_metric_unpopularity,
)


class DummyTrainDataset:
    def __init__(
        self,
        item_popularity: Sequence[int] = (0, 5, 3, 1),
        train_item_popularity: Sequence[int] | None = None,
        item_size: int | None = None,
    ) -> None:
        self.item_popularity = np.array(item_popularity, dtype=np.int64)
        if train_item_popularity is None:
            self.train_item_popularity = self.item_popularity
        else:
            self.train_item_popularity = np.array(train_item_popularity, dtype=np.int64)
        if item_size is None:
            self.item_size = len(self.item_popularity) - 1
        else:
            self.item_size = item_size


def test_calc_metric_hr_returns_mean_hit_rate():
    topk_indices = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    labels = torch.tensor([3, 7], dtype=torch.long)
    dataset = DummyTrainDataset()

    hr = calc_metric_hr(topk_indices, labels, dataset)

    assert hr["hr@3"] == pytest.approx(0.5)


def test_calc_metric_ndcg_penalizes_late_hits():
    topk_indices = torch.tensor([[8, 9, 10], [11, 12, 13]], dtype=torch.long)
    labels = torch.tensor([8, 13], dtype=torch.long)
    dataset = DummyTrainDataset()

    ndcg = calc_metric_ndcg(topk_indices, labels, dataset)

    expected_first = 1.0  # hit at rank 1
    expected_second = 1.0 / np.log2(4)  # hit at rank 3
    expected_mean = (expected_first + expected_second) / 2.0
    assert ndcg["ndcg@3"] == pytest.approx(expected_mean)


def test_calc_metric_popularity_respects_thresholds():
    topk_indices = torch.tensor([[1, 2], [3, 1]], dtype=torch.long)
    labels = torch.zeros(2, dtype=torch.long)
    dataset = DummyTrainDataset(item_popularity=(0, 10, 5, 1))

    popularity = calc_metric_popularity(topk_indices, labels, dataset, p=(0.25, 0.75))

    assert popularity["popularity@2-0.25"] == pytest.approx(0.5)
    assert popularity["popularity@2-0.75"] == pytest.approx(0.75)


def test_calc_metric_unpopularity_counts_rare_items():
    topk_indices = torch.tensor([[1, 2], [3, 2]], dtype=torch.long)
    labels = torch.zeros(2, dtype=torch.long)
    dataset = DummyTrainDataset(item_popularity=(0, 1, 5, 10))

    unpopularity = calc_metric_unpopularity(topk_indices, labels, dataset, p=(0.34, 0.67))

    assert unpopularity["unpopularity@2-0.34"] == pytest.approx(0.25)
    assert unpopularity["unpopularity@2-0.67"] == pytest.approx(0.75)


def test_calc_metric_arp_returns_mean_popularity():
    # items: [0]=pad, [1]=10, [2]=20, [3]=30
    topk_indices = torch.tensor([[1, 2, 3], [1, 1, 2]], dtype=torch.long)
    labels = torch.zeros(2, dtype=torch.long)
    dataset = DummyTrainDataset(train_item_popularity=(0, 10, 20, 30))

    result = calc_metric_arp(topk_indices, labels, dataset, target_k=3)

    # All recommended items: [1,2,3,1,1,2] -> popularities: [10,20,30,10,10,20]
    # Mean = (10+20+30+10+10+20) / 6 = 100/6
    assert result["arp@3"] == pytest.approx(100.0 / 6.0)


def test_calc_metric_arp_returns_empty_when_k_mismatches():
    topk_indices = torch.tensor([[1, 2, 3]], dtype=torch.long)
    labels = torch.zeros(1, dtype=torch.long)
    dataset = DummyTrainDataset(train_item_popularity=(0, 5, 3, 1))

    result = calc_metric_arp(topk_indices, labels, dataset, target_k=5)

    assert result == {}


def test_calc_metric_arp_uniform_popularity():
    topk_indices = torch.tensor([[1, 2], [3, 1]], dtype=torch.long)
    labels = torch.zeros(2, dtype=torch.long)
    dataset = DummyTrainDataset(train_item_popularity=(0, 7, 7, 7))

    result = calc_metric_arp(topk_indices, labels, dataset, target_k=2)

    assert result["arp@2"] == pytest.approx(7.0)


def test_calc_metric_gini_perfect_equality():
    # Each item (1,2,3) recommended exactly twice -> uniform exposure
    topk_indices = torch.tensor([[1, 2], [3, 1], [2, 3]], dtype=torch.long)
    labels = torch.zeros(3, dtype=torch.long)
    dataset = DummyTrainDataset(item_popularity=(0, 5, 5, 5), item_size=3)

    result = calc_metric_gini(topk_indices, labels, dataset, target_k=2)

    assert result["gini@2"] == pytest.approx(0.0)


def test_calc_metric_gini_high_inequality():
    # Only item 1 is recommended, items 2-4 have zero exposure
    topk_indices = torch.tensor([[1, 1], [1, 1]], dtype=torch.long)
    labels = torch.zeros(2, dtype=torch.long)
    dataset = DummyTrainDataset(item_popularity=(0, 5, 5, 5, 5), item_size=4)

    result = calc_metric_gini(topk_indices, labels, dataset, target_k=2)

    # exposure_counts = [4, 0, 0, 0] sorted -> [0, 0, 0, 4]
    # n=4, total=4, ranks=[1,2,3,4]
    # gini = ((2*1-4-1)*0 + (2*2-4-1)*0 + (2*3-4-1)*0 + (2*4-4-1)*4) / (4*4)
    #      = ((-3)*0 + (-1)*0 + 1*0 + 3*4) / 16 = 12/16 = 0.75
    assert result["gini@2"] == pytest.approx(0.75)


def test_calc_metric_gini_partial_inequality():
    # Items: 1 appears 3 times, 2 appears 1 time, 3 appears 0 times
    topk_indices = torch.tensor([[1, 1], [1, 2]], dtype=torch.long)
    labels = torch.zeros(2, dtype=torch.long)
    dataset = DummyTrainDataset(item_popularity=(0, 5, 5, 5), item_size=3)

    result = calc_metric_gini(topk_indices, labels, dataset, target_k=2)

    # exposure_counts = [3, 1, 0] sorted -> [0, 1, 3]
    # n=3, total=4, ranks=[1,2,3]
    # gini = ((2*1-3-1)*0 + (2*2-3-1)*1 + (2*3-3-1)*3) / (3*4)
    #      = ((-2)*0 + 0*1 + 2*3) / 12 = 6/12 = 0.5
    assert result["gini@2"] == pytest.approx(0.5)


def test_calc_metric_gini_returns_empty_when_k_mismatches():
    topk_indices = torch.tensor([[1, 2, 3]], dtype=torch.long)
    labels = torch.zeros(1, dtype=torch.long)
    dataset = DummyTrainDataset(item_size=3)

    result = calc_metric_gini(topk_indices, labels, dataset, target_k=5)

    assert result == {}


def test_calc_metric_gini_zero_exposure():
    topk_indices = torch.zeros((0, 5), dtype=torch.long)
    labels = torch.zeros(0, dtype=torch.long)
    dataset = DummyTrainDataset(item_popularity=(0, 1, 2, 3), item_size=3)

    result = calc_metric_gini(topk_indices, labels, dataset, target_k=5)

    assert result["gini@5"] == pytest.approx(0.0)


def test_metric_factory_returns_registered_metric():
    hr_metric = SeqRecMetricFactory.create("hr")

    assert hr_metric is calc_metric_hr


def test_metric_factory_raises_for_unknown_metric():
    with pytest.raises(ValueError):
        SeqRecMetricFactory.create("unknown")
