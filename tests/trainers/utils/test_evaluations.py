from typing import Sequence

import numpy as np
import pytest
import torch

from genrec.trainers.utils.evaluations import (
    MetricFactory,
    calc_metric_hr,
    calc_metric_ndcg,
    calc_metric_popularity,
    calc_metric_unpopularity,
)


class DummyTrainDataset:
    def __init__(self, item_popularity: Sequence[int] = (0, 5, 3, 1)) -> None:
        self.item_popularity = np.array(item_popularity, dtype=np.int64)


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


def test_metric_factory_returns_registered_metric():
    hr_metric = MetricFactory.create("hr")

    assert hr_metric is calc_metric_hr


def test_metric_factory_raises_for_unknown_metric():
    with pytest.raises(ValueError):
        MetricFactory.create("unknown")
