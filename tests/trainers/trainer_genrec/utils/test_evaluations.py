from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pytest
import torch
from transformers.trainer_utils import EvalPrediction

from genrec.trainers.trainer_genrec.utils.evaluations import (
    GenRecMetricFactory,
    batch_sid_to_item,
    calc_metric_hr,
    calc_metric_ndcg,
    calc_metric_popularity,
    calc_metric_unpopularity,
    clip_top_k,
    compute_genrec_metrics,
)


SID_MAPPING: Dict[Tuple[int, int], int] = {
    (1, 11): 1,
    (2, 22): 2,
    (3, 33): 3,
    (4, 44): 4,
}


@dataclass
class DummyGenRecDataset:
    sid2item: Dict[Tuple[int, ...], int]
    item_popularity: np.ndarray


def _make_dataset(popularity: Tuple[int, ...]) -> DummyGenRecDataset:
    popularity_array = np.array(popularity, dtype=np.int64)
    return DummyGenRecDataset(sid2item=dict(SID_MAPPING), item_popularity=popularity_array)


def test_batch_sid_to_item_maps_sequences_to_item_ids():
    sids = torch.tensor(
        [
            [[1, 11], [9, 99]],
            [[2, 22], [3, 33]],
        ],
        dtype=torch.long,
    )

    item_ids = batch_sid_to_item(sids, SID_MAPPING)

    expected = torch.tensor([[1, 0], [2, 3]], dtype=torch.long)
    assert torch.equal(item_ids, expected)


def test_calc_metric_hr_counts_exact_sequence_matches():
    topk_sids = torch.tensor(
        [
            [[1, 11], [2, 22]],
            [[3, 33], [4, 44]],
        ],
        dtype=torch.long,
    )
    labels = torch.tensor([[1, 11], [9, 99]], dtype=torch.long)
    dataset = _make_dataset((0, 1, 1, 1, 1))

    result = calc_metric_hr(topk_sids, labels, dataset)

    assert result["hr@2"] == pytest.approx(0.25)


def test_calc_metric_ndcg_downweights_late_hits():
    topk_sids = torch.tensor(
        [
            [[1, 11], [2, 22]],
            [[2, 22], [3, 33]],
        ],
        dtype=torch.long,
    )
    labels = torch.tensor([[1, 11], [3, 33]], dtype=torch.long)
    dataset = _make_dataset((0, 1, 1, 1, 1))

    result = calc_metric_ndcg(topk_sids, labels, dataset)

    discounts = torch.tensor([1.0, 1.0 / torch.log2(torch.tensor(3.0))])
    expected_first = discounts[0]
    expected_second = discounts[1]
    expected_mean = ((expected_first + expected_second) / 2).item()
    assert result["ndcg@2"] == pytest.approx(expected_mean)


def test_calc_metric_popularity_respects_popularity_cutoffs():
    topk_sids = torch.tensor(
        [
            [[1, 11], [2, 22]],
            [[3, 33], [1, 11]],
        ],
        dtype=torch.long,
    )
    dataset = _make_dataset((0, 30, 20, 10, 5))

    result = calc_metric_popularity(topk_sids, torch.zeros((2, 2), dtype=torch.long), dataset, p=(1 / 3, 2 / 3))

    assert result["popularity@2-0.3333333333333333"] == pytest.approx(0.5)
    assert result["popularity@2-0.6666666666666666"] == pytest.approx(0.75)


def test_calc_metric_unpopularity_counts_tail_predictions():
    topk_sids = torch.tensor(
        [
            [[3, 33], [1, 11]],
            [[3, 33], [2, 22]],
        ],
        dtype=torch.long,
    )
    dataset = _make_dataset((0, 20, 10, 1, 5))

    result = calc_metric_unpopularity(topk_sids, torch.zeros((2, 2), dtype=torch.long), dataset, p=(1 / 3, 2 / 3))

    assert result["unpopularity@2-0.3333333333333333"] == pytest.approx(0.5)
    assert result["unpopularity@2-0.6666666666666666"] == pytest.approx(0.5)


def test_clip_top_k_truncates_and_deduplicates():
    clipped = clip_top_k((1, 5, 5, 10), item_size=3)

    assert clipped == (1, 3)


def test_genrec_metric_factory_returns_registered_metric():
    metric = GenRecMetricFactory.create("hr")

    assert metric is calc_metric_hr


def test_genrec_metric_factory_errors_on_unknown_metric():
    with pytest.raises(ValueError):
        GenRecMetricFactory.create("unknown")


def test_compute_genrec_metrics_respects_top_k_and_metrics():
    preds = np.array(
        [
            [[1, 11], [2, 22]],
            [[2, 22], [1, 11]],
        ],
        dtype=np.int64,
    )
    labels = np.array(
        [
            [1, 11],
            [2, 22],
        ],
        dtype=np.int64,
    )
    prediction = EvalPrediction(predictions=preds, label_ids=labels)
    dataset = _make_dataset((0, 3, 2, 1, 1))

    metrics = compute_genrec_metrics(prediction, dataset, top_k=(1, 2), metrics=(("hr", {}), ("ndcg", {})))

    assert metrics["hr@1"] == pytest.approx(1.0)
    assert metrics["hr@2"] == pytest.approx(0.5)
    assert metrics["ndcg@1"] == pytest.approx(1.0)
    assert metrics["ndcg@2"] == pytest.approx(1.0)
