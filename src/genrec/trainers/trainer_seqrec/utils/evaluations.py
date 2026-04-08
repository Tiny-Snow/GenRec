"""Evaluation utilities for seqrec models."""

from __future__ import annotations

from typing import Any, Callable, Dict, Protocol, Sequence, Tuple, Union

from jaxtyping import Int
import numpy as np
import torch
from transformers import EvalPrediction

from ....datasets import SeqRecDataset

__all__ = [
    "SeqRecMetricFactory",
    "SeqRecMetricFn",
    "calc_metric_hr",
    "calc_metric_ndcg",
    "calc_metric_arp",
    "calc_metric_gini",
    "calc_metric_popularity",
    "calc_metric_unpopularity",
    "clip_top_k",
    "compute_seqrec_metrics",
]


def compute_seqrec_metrics(
    prediction: EvalPrediction,
    train_dataset: SeqRecDataset,
    top_k: Sequence[int] = (1, 5, 10),
    metrics: Sequence[Tuple[str, Dict[str, Any]]] = (
        ("hr", {}),
        ("ndcg", {}),
        ("popularity", {"p": (0.1, 0.2)}),
        ("unpopularity", {"p": (0.2, 0.4)}),
    ),
    device: Union[torch.device, str, None] = None,
) -> Dict[str, float]:
    """Compute metrics for sequential recommendation tasks.

    Args:
        prediction (EvalPrediction): Object containing model predictions and labels. Predictions are
            expected to be the precomputed top-k item indices per user (shape: ``[num_users, max_k]``).
        train_dataset (SeqRecDataset): Dataset used during training; required for global metrics
            such as popularity-based measurements.
        top_k (Sequence[int]): Cutoff values for computing top-K metrics, determining how many
            predictions to consider for each metric. Default is (1, 5, 10).
        metrics (Sequence[Tuple[str, Dict[str, Any]]]): Metric specifications, where each tuple
            comprises the metric name and an optional parameter dictionary. Default is
            [('hr', {}), ('ndcg', {}), ('popularity', {'p': (0.1, 0.2)}),
            ('unpopularity', {'p': (0.2, 0.4)})].
        device (Union[torch.device, str, None]): Device used for metric computations.
            If None, defaults to CPU. Default is None.

    Returns:
        Dict[str, float]: Dictionary containing computed metric values keyed by metric name.

    .. note::
        As we may call this evaluation function for global metrics (e.g., popularity/fairness),
        you should ensure the `train_dataset` is provided if any global metrics are specified.
        In addition, `batch_eval_metrics` in `SeqRecTrainingArguments` should be set to `False`
        to avoid conflicts.
    """
    torch_device = torch.device(device) if device is not None else torch.device("cpu")

    topk_indices: Int[torch.Tensor, "B K"]
    if isinstance(prediction.predictions, tuple):  # pragma: no cover - rarely used
        topk_indices = torch.as_tensor(prediction.predictions[0], dtype=torch.long, device=torch_device)
    else:
        topk_indices = torch.as_tensor(prediction.predictions, dtype=torch.long, device=torch_device)

    labels: Int[np.ndarray, "B L"]
    if isinstance(prediction.label_ids, tuple):  # pragma: no cover - rarely used
        labels = prediction.label_ids[0].astype(np.int64)
    else:
        labels = prediction.label_ids.astype(np.int64)
    last_step_labels: Int[torch.Tensor, "B"]
    last_step_labels = torch.as_tensor(labels[:, -1], dtype=torch.long, device=torch_device)

    results: Dict[str, float] = {}
    for k in top_k:
        sliced_topk_indices = topk_indices[:, :k]
        for metric_name, metric_params in metrics:
            metric_fn = SeqRecMetricFactory.create(metric_name)
            metric_results = metric_fn(
                topk_indices=sliced_topk_indices,
                labels=last_step_labels,
                train_dataset=train_dataset,
                **metric_params,
            )
            results.update(metric_results)

    return results


def clip_top_k(top_k: Sequence[int], item_size: int) -> Tuple[int, ...]:
    """Clamp sorted ``top_k`` cutoffs to ``item_size``, sort and remove duplicates if any."""
    top_k_set = set([min(k, item_size) for k in top_k])
    return tuple(sorted(top_k_set))


class SeqRecMetricFn(Protocol):  # pragma: no cover - protocol
    """Protocol for seqrec metric functions."""

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


class SeqRecMetricFactory:  # pragma: no cover - factory class
    """Factory for creating seqrec metric functions."""

    _registry: dict[str, SeqRecMetricFn] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[SeqRecMetricFn], SeqRecMetricFn]:
        """Decorator to register a metric function."""

        def decorator(fn: SeqRecMetricFn) -> SeqRecMetricFn:
            if name in cls._registry:
                raise ValueError(f"Metric '{name}' is already registered.")
            cls._registry[name] = fn
            return fn

        return decorator

    @classmethod
    def create(cls, name: str) -> SeqRecMetricFn:
        """Create a metric function by name."""
        if name not in cls._registry:
            raise ValueError(f"Metric '{name}' is not registered.")
        metric_fn = cls._registry[name]
        return metric_fn


@SeqRecMetricFactory.register("hr")
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


@SeqRecMetricFactory.register("ndcg")
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


@SeqRecMetricFactory.register("popularity")
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


@SeqRecMetricFactory.register("arp")
def calc_metric_arp(
    topk_indices: Int[torch.Tensor, "B K"],
    labels: Int[torch.Tensor, "B"],
    train_dataset: SeqRecDataset,
    target_k: int = 5,
    **kwargs: Any,
) -> Dict[str, float]:
    """Calculate Average Recommendation Popularity (ARP) at a fixed cutoff.

    ARP is computed as the mean training popularity of all recommended items across
    all users and all recommendation positions. Since :func:`compute_seqrec_metrics`
    aggregates predictions over the full evaluation set before invoking metric
    functions, this value is naturally computed across batches.

    Args:
        topk_indices (Int[torch.Tensor, "B K"]): Tensor containing top-K predicted item indices.
        labels (Int[torch.Tensor, "B"]): Ground-truth labels (unused, kept for uniform signature).
        train_dataset (SeqRecDataset): Training dataset used to obtain item popularity counts.
        target_k (int): Fixed cutoff at which to report ARP. Defaults to 5.

    Returns:
        Dict[str, float]: Dictionary mapping ``"arp@5"`` to its float value when ``K == target_k``;
            otherwise an empty dictionary.
    """
    del labels  # unused

    K = topk_indices.shape[1]
    if K != target_k:
        return {}

    item_popularity = torch.as_tensor(train_dataset.train_item_popularity, device=topk_indices.device, dtype=torch.float32)
    recommended_popularity = item_popularity[topk_indices]
    arp = recommended_popularity.mean().item()
    return {f"arp@{K}": arp}


@SeqRecMetricFactory.register("gini")
def calc_metric_gini(
    topk_indices: Int[torch.Tensor, "B K"],
    labels: Int[torch.Tensor, "B"],
    train_dataset: SeqRecDataset,
    target_k: int = 5,
    **kwargs: Any,
) -> Dict[str, float]:
    """Calculate the Gini coefficient of item exposure counts at a fixed cutoff.

    Let ``x_i`` denote the number of times item ``i`` appears in the top-K
    recommendations over the entire evaluation set. The Gini coefficient is:

    .. math::
        \\mathrm{Gini} = \\frac{\\sum_i \\sum_j |x_i - x_j|}{2n \\sum_i x_i}

    where ``n`` is the number of non-padding items. We compute an equivalent
    sorted-vector formulation for efficiency while still aggregating exposure
    counts across the whole evaluation set.

    Args:
        topk_indices (Int[torch.Tensor, "B K"]): Tensor containing top-K predicted item indices.
        labels (Int[torch.Tensor, "B"]): Ground-truth labels (unused, kept for uniform signature).
        train_dataset (SeqRecDataset): Training dataset used only to infer catalogue size.
        target_k (int): Fixed cutoff at which to report Gini. Defaults to 5.

    Returns:
        Dict[str, float]: Dictionary mapping ``"gini@5"`` to its float value when ``K == target_k``;
            otherwise an empty dictionary.
    """
    del labels  # unused

    K = topk_indices.shape[1]
    if K != target_k:
        return {}

    item_size = train_dataset.item_size
    exposure_counts = torch.bincount(topk_indices.reshape(-1), minlength=item_size + 1)[1:].to(torch.float32)
    total_exposure = exposure_counts.sum()
    if total_exposure <= 0:
        return {f"gini@{K}": 0.0}

    sorted_exposure, _ = torch.sort(exposure_counts)
    num_items = sorted_exposure.numel()
    ranks = torch.arange(1, num_items + 1, device=sorted_exposure.device, dtype=torch.float32)
    gini = ((2 * ranks - num_items - 1) * sorted_exposure).sum() / (num_items * total_exposure)
    return {f"gini@{K}": gini.item()}


@SeqRecMetricFactory.register("unpopularity")
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
