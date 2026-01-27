from __future__ import annotations

import numpy as np
import pytest
from transformers import EvalPrediction

from genrec.trainers.trainer_quantizer.utils.evaluations import (
    QuantizerMetricFactory,
    calc_metric_code_collision,
    calc_metric_codebook_usage,
    compute_quantizer_metrics,
)


class DummyTrainDataset:
    item_popularity = np.ones(5, dtype=np.int64)


def test_compute_quantizer_metrics_returns_expected_values() -> None:
    semantic_ids = np.array(
        [
            [0, 1],
            [0, 1],
            [2, 3],
        ],
        dtype=np.int64,
    )
    reconstruction_loss = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    codebook_loss = np.array([0.5, 1.5, 2.5], dtype=np.float32)
    commitment_loss = np.array([2.0, 0.0, 1.0], dtype=np.float32)
    item_id = np.array([1, 2, 3], dtype=np.int64)

    prediction = EvalPrediction(
        predictions=(semantic_ids, reconstruction_loss, codebook_loss, commitment_loss, item_id),
        label_ids=None,
    )

    metrics = compute_quantizer_metrics(
        prediction,
        train_dataset=DummyTrainDataset(),
        codebook_size=4,
    )

    assert metrics["reconstruction_loss"] == pytest.approx(2.0)
    assert metrics["codebook_loss"] == pytest.approx(1.5)
    assert metrics["commitment_loss"] == pytest.approx(1.0)
    assert metrics["codebook_0_usage"] == pytest.approx(0.5)
    assert metrics["codebook_1_usage"] == pytest.approx(0.5)
    assert metrics["code_collision_rate"] == pytest.approx(1.0 / 3.0)


def test_compute_quantizer_metrics_requires_tuple_predictions() -> None:
    prediction = EvalPrediction(predictions=np.zeros((2, 2)), label_ids=np.zeros(2))

    with pytest.raises(ValueError, match="Predictions should be a tuple"):
        compute_quantizer_metrics(
            prediction,
            train_dataset=DummyTrainDataset(),
            codebook_size=4,
        )


def test_metric_functions_directly() -> None:
    semantic_ids = np.array(
        [
            [1, 1],
            [1, 2],
        ],
        dtype=np.int64,
    )
    item_id = np.array([1, 2], dtype=np.int64)

    usage = calc_metric_codebook_usage(
        semantic_ids=semantic_ids,
        item_id=item_id,
        train_dataset=DummyTrainDataset(),
        codebook_size=4,
    )
    collision = calc_metric_code_collision(
        semantic_ids=semantic_ids,
        item_id=item_id,
        train_dataset=DummyTrainDataset(),
        codebook_size=4,
    )

    assert usage["codebook_0_usage"] == pytest.approx(1.0 / 4.0)
    assert usage["codebook_1_usage"] == pytest.approx(2.0 / 4.0)
    assert collision["code_collision_rate"] == pytest.approx(0.0)


def test_metric_factory_raises_for_unknown_metric() -> None:
    with pytest.raises(ValueError, match="not registered"):
        QuantizerMetricFactory.create("missing_metric")
