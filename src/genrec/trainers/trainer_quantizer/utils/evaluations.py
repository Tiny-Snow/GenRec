"""Evaluation utilities for quantizer trainers."""

from __future__ import annotations

from typing import Any, Callable, Dict, Protocol, Sequence, Tuple

from jaxtyping import Float, Int
import numpy as np
from transformers import EvalPrediction

from ....datasets import QuantizerDataset

__all__ = [
    "QuantizerMetricFactory",
    "QuantizerMetricFn",
    "calc_metric_codebook_usage",
    "calc_metric_code_collision",
    "compute_quantizer_metrics",
]


def compute_quantizer_metrics(
    prediction: EvalPrediction,
    train_dataset: QuantizerDataset,
    codebook_size: int,
    metrics: Sequence[Tuple[str, Dict[str, Any]]] = (
        ("codebook_usage", {}),
        ("code_collision", {}),
    ),
) -> Dict[str, float]:
    """Compute metrics for quantizer trainers.

    Args:
        prediction (EvalPrediction): Object containing model predictions and labels. Predictions are
            expected to be the dict values from `QuantizerTrainer.compute_loss`'s output dict,
            at least including `semantic_ids`, `reconstruction_loss`, `codebook_loss`, `commitment_loss`,
            and `item_id` in the first 5 elements. The labels are ignored for quantizer metrics.
        train_dataset (QuantizerDataset): Dataset used during training; required for global metrics.
        codebook_size (int): Size of the codebook used in the quantizer.
        metrics (Sequence[Tuple[str, Dict[str, Any]]]): Metric specifications, where each tuple
            comprises the metric name and an optional parameter dictionary. Default is
            [('codebook_usage', {}), ('code_collision', {})].

    Returns:
        Dict[str, float]: Dictionary containing computed metric values keyed by metric name.
    """
    if isinstance(prediction.predictions, tuple):
        assert len(prediction.predictions) >= 5, (
            "Predictions should contain at least 5 elements: "
            "`semantic_ids`, `reconstruction_loss`, `codebook_loss`, `commitment_loss`, and `item_id`."
        )
        semantic_ids: Int[np.ndarray, "B C"] = prediction.predictions[0]
        reconstruction_loss: Float[np.ndarray, "B"] = prediction.predictions[1]
        codebook_loss: Float[np.ndarray, "B"] = prediction.predictions[2]
        commitment_loss: Float[np.ndarray, "B"] = prediction.predictions[3]
        item_id: Int[np.ndarray, "B"] = prediction.predictions[4]
    else:
        raise ValueError("Predictions should be a tuple containing model output dict's values.")

    results: Dict[str, float] = {
        "reconstruction_loss": float(np.mean(reconstruction_loss)),
        "codebook_loss": float(np.mean(codebook_loss)),
        "commitment_loss": float(np.mean(commitment_loss)),
    }
    for metric_name, metric_params in metrics:
        metric_fn = QuantizerMetricFactory.create(metric_name)
        metric_results = metric_fn(
            semantic_ids=semantic_ids,
            item_id=item_id,
            train_dataset=train_dataset,
            codebook_size=codebook_size,
            **metric_params,
        )
        results.update(metric_results)

    return results


class QuantizerMetricFn(Protocol):
    """Protocol for quantizer metric functions."""

    def __call__(  # pragma: no cover - protocol
        self,
        semantic_ids: Int[np.ndarray, "B C"],
        item_id: Int[np.ndarray, "B"],
        train_dataset: QuantizerDataset,
        codebook_size: int,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Compute the metric.

        Args:
            semantic_ids (Int[np.ndarray, "B C"]): Semantic IDs predicted by the quantizer.
            item_id (Int[np.ndarray, "B"]): Item IDs corresponding to the input embeddings.
            train_dataset (QuantizerDataset): Dataset used during training; required for global metrics.
            codebook_size (int): Size of the codebook used in the quantizer.
            **kwargs (Any): Additional keyword arguments for metric computation.

        Returns:
            Dict[str, float]: Dictionary containing computed metric values keyed by metric name.
        """
        ...


class QuantizerMetricFactory:  # pragma: no cover - factory class
    """Factory for creating quantizer metric functions."""

    _registry: dict[str, QuantizerMetricFn] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[QuantizerMetricFn], QuantizerMetricFn]:
        """Decorator to register a metric function."""

        def decorator(fn: QuantizerMetricFn) -> QuantizerMetricFn:
            if name in cls._registry:
                raise ValueError(f"Metric '{name}' is already registered.")
            cls._registry[name] = fn
            return fn

        return decorator

    @classmethod
    def create(cls, name: str) -> QuantizerMetricFn:
        """Create a metric function by name."""
        if name not in cls._registry:
            raise ValueError(f"Metric '{name}' is not registered.")
        metric_fn = cls._registry[name]
        return metric_fn


@QuantizerMetricFactory.register("codebook_usage")
def calc_metric_codebook_usage(
    semantic_ids: Int[np.ndarray, "B C"],
    item_id: Int[np.ndarray, "B"],
    train_dataset: QuantizerDataset,
    codebook_size: int,
    **kwargs: Any,
) -> Dict[str, float]:
    """Calculate codebook usage metric.

    Args:
        semantic_ids (Int[np.ndarray, "B C"]): Semantic IDs predicted by the quantizer.
        item_id (Int[np.ndarray, "B"]): Item IDs corresponding to the input embeddings.
        train_dataset (QuantizerDataset): Dataset used during training; required for global metrics.
        codebook_size (int): Size of the codebook used in the quantizer.
        **kwargs (Any): Additional keyword arguments for metric computation.

    Returns:
        Dict[str, float]: Dictionary containing the codebook usage metric.
    """
    num_codebooks = semantic_ids.shape[1]
    results: Dict[str, float] = {}
    for codebook_idx in range(num_codebooks):
        used_codes = np.unique(semantic_ids[:, codebook_idx])
        usage_ratio = len(used_codes) / codebook_size
        results[f"codebook_{codebook_idx}_usage"] = usage_ratio
    return results


@QuantizerMetricFactory.register("code_collision")
def calc_metric_code_collision(
    semantic_ids: Int[np.ndarray, "B C"],
    item_id: Int[np.ndarray, "B"],
    train_dataset: QuantizerDataset,
    codebook_size: int,
    **kwargs: Any,
) -> Dict[str, float]:
    """Calculate code collision metric.

    Args:
        semantic_ids (Int[np.ndarray, "B C"]): Semantic IDs predicted by the quantizer.
        item_id (Int[np.ndarray, "B"]): Item IDs corresponding to the input embeddings.
        train_dataset (QuantizerDataset): Dataset used during training; required for global metrics.
        codebook_size (int): Size of the codebook used in the quantizer.
        **kwargs (Any): Additional keyword arguments for metric computation.

    Returns:
        Dict[str, float]: Dictionary containing the code collision metric.
    """
    codes = [tuple(code_ids) for code_ids in semantic_ids]
    unique_codes = set(codes)
    num_collisions = len(codes) - len(unique_codes)
    collision_rate = num_collisions / len(codes) if len(codes) > 0 else 0.0
    return {"code_collision_rate": collision_rate}
