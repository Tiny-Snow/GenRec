"""Evaluation utilities for recommendation tasks."""

from __future__ import annotations

from typing import Any, Callable, Dict, Protocol, Sequence

import torch
from jaxtyping import Int

from ...datasets import SeqRecDataset

__all__ = [
    "calc_metric_hr",
    "calc_metric_ndcg",
    "calc_metric_popularity",
    "calc_metric_unpopularity",
    "MetricFactory",
    "MetricFn",
]


class MetricFn(Protocol):  # pragma: no cover - protocol
    """Protocol for metric functions."""

    def __call__(
        self,
        topk_indices: Int[torch.Tensor, "B K"],
        labels: Int[torch.Tensor, "B"],
        train_dataset: SeqRecDataset,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Calculate metric values.

        Args:
            topk_indices (Int[torch.Tensor, "B K"]): Tensor containing top-K predicted item indices
                for each user, sorted in descending order of predicted relevance.
            labels (Int[torch.Tensor, "B"]): Tensor containing ground-truth item indices for each user.
            train_dataset (SeqRecDataset): Training dataset used to compute item-level statistics.
            **kwargs (Any): Additional keyword arguments for specific metric calculations.

        Returns:
            Dict[str, float]: Mapping from metric names to their computed values.
        """
        ...


class MetricFactory:  # pragma: no cover - factory class
    """Factory for creating metric functions."""

    _registry: dict[str, MetricFn] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[MetricFn], MetricFn]:
        """Decorator to register a metric function."""

        def decorator(fn: MetricFn) -> MetricFn:
            if name in cls._registry:
                raise ValueError(f"Metric '{name}' is already registered.")
            cls._registry[name] = fn
            return fn

        return decorator

    @classmethod
    def create(cls, name: str) -> MetricFn:
        """Create a metric function by name."""
        if name not in cls._registry:
            raise ValueError(f"Metric '{name}' is not registered.")
        metric_fn = cls._registry[name]
        return metric_fn


@MetricFactory.register("hr")
def calc_metric_hr(
    topk_indices: Int[torch.Tensor, "B K"],
    labels: Int[torch.Tensor, "B"],
    train_dataset: SeqRecDataset,
    **kwargs: Any,
) -> Dict[str, float]:
    """Calculate Hit Rate (HR) metric.

    Args:
        topk_indices (Int[torch.Tensor, "B K"]): Tensor containing top-K predicted item indices for each user.
        labels (Int[torch.Tensor, "B"]): Tensor containing ground-truth item indices for each user.
        train_dataset (SeqRecDataset): Training dataset used to compute item popularity (unused for HR but kept
            for uniform signature).

    Returns:
        Dict[str, float]: Dictionary mapping "hr@K" to its float value.
    """
    K = topk_indices.shape[1]
    hits = torch.any(topk_indices == labels.unsqueeze(1), dim=1)
    hr = hits.float().mean().item()
    return {f"hr@{K}": hr}


@MetricFactory.register("ndcg")
def calc_metric_ndcg(
    topk_indices: Int[torch.Tensor, "B K"],
    labels: Int[torch.Tensor, "B"],
    train_dataset: SeqRecDataset,
    **kwargs: Any,
) -> Dict[str, float]:
    """Calculate Normalized Discounted Cumulative Gain (NDCG) metric.

    Args:
        topk_indices (Int[torch.Tensor, "B K"]): Tensor containing top-K predicted item indices for each user.
        labels (Int[torch.Tensor, "B"]): Tensor containing ground-truth item indices for each user.
        train_dataset (SeqRecDataset): Training dataset used to compute item popularity (unused for NDCG but
            kept for uniform signature).

    Returns:
        Dict[str, float]: Dictionary mapping "ndcg@K" to its float value.
    """
    K = topk_indices.shape[1]
    relevance = (topk_indices == labels.unsqueeze(1)).float()
    discounts = 1.0 / torch.log2(torch.arange(2, K + 2, device=topk_indices.device, dtype=torch.float32))
    dcg = (relevance * discounts.view(1, -1)).sum(dim=1)
    ndcg = dcg.mean().item()
    return {f"ndcg@{K}": ndcg}


@MetricFactory.register("popularity")
def calc_metric_popularity(
    topk_indices: Int[torch.Tensor, "B K"],
    labels: Int[torch.Tensor, "B"],
    train_dataset: SeqRecDataset,
    p: Sequence[float] = (0.1, 0.2),
    **kwargs: Any,
) -> Dict[str, float]:
    """Calculate Top-p Popularity metric.

    Args:
        topk_indices (Int[torch.Tensor, "B K"]): Tensor containing top-K predicted item indices for each user.
        labels (Int[torch.Tensor, "B"]): Tensor containing ground truth item indices for each user.
        train_dataset (SeqRecDataset): Training dataset used to derive popularity statistics.
        p (Sequence[float]): Popularity thresholds (between 0 and 1) defining popular items.

    Returns:
        Dict[str, float]: Dictionary mapping "popularity@K-p" to its float value.
    """
    device = topk_indices.device
    B, K = topk_indices.shape

    item_popularity = torch.as_tensor(train_dataset.item_popularity, device=device, dtype=torch.float32)
    I = item_popularity.shape[0] - 1
    pred_cnt = torch.bincount(topk_indices.reshape(-1), minlength=I + 1)
    pred_sum = float(B * K)
    popularity_rank = torch.argsort(item_popularity[1:], descending=True)

    results: Dict[str, float] = {}
    for threshold in p:
        popular_threshold = max(1, int(I * threshold))
        selected_items = popularity_rank[:popular_threshold]
        pop_sum = pred_cnt.index_select(0, selected_items + 1).sum().float()
        results[f"popularity@{K}-{threshold}"] = (pop_sum / pred_sum).item()

    return results


@MetricFactory.register("unpopularity")
def calc_metric_unpopularity(
    topk_indices: Int[torch.Tensor, "B K"],
    labels: Int[torch.Tensor, "B"],
    train_dataset: SeqRecDataset,
    p: Sequence[float] = (0.1, 0.2),
    **kwargs: Any,
) -> Dict[str, float]:
    """Calculate Bottom-p Unpopularity metric.

    Args:
        topk_indices (Int[torch.Tensor, "B K"]): Tensor containing top-K predicted item indices for each user.
        labels (Int[torch.Tensor, "B"]): Tensor containing ground truth item indices for each user.
        train_dataset (SeqRecDataset): Training dataset used to derive popularity statistics.
        p (Sequence[float]): Unpopularity thresholds (between 0 and 1) defining unpopular items.

    Returns:
        Dict[str, float]: Dictionary mapping "unpopularity@K-p" to its float value.
    """
    device = topk_indices.device
    B, K = topk_indices.shape

    item_popularity = torch.as_tensor(train_dataset.item_popularity, device=device, dtype=torch.float32)
    I = item_popularity.shape[0] - 1
    pred_cnt = torch.bincount(topk_indices.reshape(-1), minlength=I + 1)
    pred_sum = float(B * K)
    unpopularity_rank = torch.argsort(item_popularity[1:], descending=False)

    results: Dict[str, float] = {}
    for threshold in p:
        unpopular_threshold = max(1, int(I * threshold))
        selected_items = unpopularity_rank[:unpopular_threshold]
        unpop_sum = pred_cnt.index_select(0, selected_items + 1).sum().float()
        results[f"unpopularity@{K}-{threshold}"] = (unpop_sum / pred_sum).item()

    return results
