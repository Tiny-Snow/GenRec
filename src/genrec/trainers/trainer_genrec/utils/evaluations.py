"""Evaluation utilities for genrec models."""

from __future__ import annotations

from typing import Any, Callable, Dict, Protocol, Sequence, Tuple, Union

from jaxtyping import Int
import numpy as np
import torch
from transformers.trainer_utils import EvalPrediction

from ....datasets import GenRecDataset

__all__ = [
    "GenRecMetricFactory",
    "GenRecMetricFn",
    "calc_metric_hr",
    "calc_metric_ndcg",
    "calc_metric_popularity",
    "calc_metric_unpopularity",
    "clip_top_k",
    "compute_genrec_metrics",
]


def compute_genrec_metrics(
    prediction: EvalPrediction,
    train_dataset: GenRecDataset,
    top_k: Sequence[int] = (1, 5, 10),
    metrics: Sequence[Tuple[str, Dict[str, Any]]] = (
        ("hr", {}),
        ("ndcg", {}),
        ("popularity", {"p": (0.1, 0.2)}),
        ("unpopularity", {"p": (0.2, 0.4)}),
    ),
    device: Union[torch.device, str, None] = None,
) -> Dict[str, float]:
    """Compute metrics for generative recommendation tasks.

    Args:
        prediction (EvalPrediction): Object containing model predictions and labels. Predictions are
            expected to be the num_beams item sids per user, of shape (num_users, num_beams, sid_width).
            Labels are expected to be the ground-truth item sids of shape (num_users, sid_width).
        train_dataset (GenRecDataset): Dataset used during training; required for global metrics
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
        In addition, `batch_eval_metrics` in `GenRecTrainingArguments` should be set to `False`
        to avoid conflicts.
    """
    torch_device = torch.device(device) if device is not None else torch.device("cpu")

    beam_sequences: Int[torch.Tensor, "B num_beams C"]
    if isinstance(prediction.predictions, tuple):  # pragma: no cover - rarely used
        beam_sequences = torch.as_tensor(prediction.predictions[0], dtype=torch.long, device=torch_device)
    else:
        beam_sequences = torch.as_tensor(prediction.predictions, dtype=torch.long, device=torch_device)

    labels: Int[torch.Tensor, "B C"]
    if isinstance(prediction.label_ids, tuple):  # pragma: no cover - rarely used
        labels = torch.as_tensor(prediction.label_ids[0], dtype=torch.long, device=torch_device)
    else:
        labels = torch.as_tensor(prediction.label_ids, dtype=torch.long, device=torch_device)

    num_beams = beam_sequences.size(1)
    assert num_beams >= max(top_k), f"Number of beams ({num_beams}) < max top_k ({max(top_k)})."

    results: Dict[str, float] = {}
    for k in top_k:
        sliced_beam_seqs = beam_sequences[:, :k, :]
        for metric_name, metric_params in metrics:
            metric_fn = GenRecMetricFactory.create(metric_name)
            metric_results = metric_fn(
                topk_sids=sliced_beam_seqs,
                labels=labels,
                train_dataset=train_dataset,
                **metric_params,
            )
            results.update(metric_results)

    return results


def clip_top_k(top_k: Sequence[int], item_size: int) -> Tuple[int, ...]:
    """Clamp sorted ``top_k`` cutoffs to ``item_size``, sort and remove duplicates if any."""
    top_k_set = set([min(k, item_size) for k in top_k])
    return tuple(sorted(top_k_set))


def batch_sid_to_item(
    sids: Int[torch.Tensor, "... C"],
    sid2item: Dict[Tuple[int, ...], int],
) -> Int[torch.Tensor, "..."]:
    """Convert a batch of SID sequences to item IDs.

    Args:
        sids (Int[torch.Tensor, "... C"]): SID sequences of shape (..., C).
        sid2item (Dict[Tuple[int, ...], int]): Mapping from SID sequences to item IDs.

    Returns:
        Int[torch.Tensor, "..."]: Corresponding item IDs of shape (...,).
    """
    sids_np = sids.cpu().numpy()
    original_shape = sids_np.shape[:-1]
    sids_reshaped = sids_np.reshape(-1, sids_np.shape[-1])
    item_ids = []
    for sid_seq in sids_reshaped:
        item_id = sid2item.get(tuple(sid_seq.tolist()), 0)  # default to padding item ID 0 if not found
        item_ids.append(item_id)
    item_ids_tensor = torch.as_tensor(item_ids, dtype=torch.long, device=sids.device)
    return item_ids_tensor.view(*original_shape)


class GenRecMetricFn(Protocol):  # pragma: no cover - protocol
    """Protocol for genrec metric functions."""

    def __call__(
        self,
        topk_sids: Int[torch.Tensor, "B K C"],
        labels: Int[torch.Tensor, "B C"],
        train_dataset: GenRecDataset,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Calculate metric values.

        Args:
            topk_sids (Int[torch.Tensor, "B K C"]): Top-K item sids predicted by the model
                sorted by their generation scores in descending order.
            labels (Int[torch.Tensor, "B C"]): Ground-truth item sids.
            train_dataset (GenRecDataset): Training dataset used to compute item-level statistics.
            **kwargs (Any): Additional keyword arguments for specific metric calculations.

        Returns:
            Dict[str, float]: Mapping from metric names to their computed values.
        """
        ...


class GenRecMetricFactory:  # pragma: no cover - factory class
    """Factory for creating genrec metric functions."""

    _registry: dict[str, GenRecMetricFn] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[GenRecMetricFn], GenRecMetricFn]:
        """Decorator to register a metric function."""

        def decorator(fn: GenRecMetricFn) -> GenRecMetricFn:
            if name in cls._registry:
                raise ValueError(f"Metric '{name}' is already registered.")
            cls._registry[name] = fn
            return fn

        return decorator

    @classmethod
    def create(cls, name: str) -> GenRecMetricFn:
        """Create a metric function by name."""
        if name not in cls._registry:
            raise ValueError(f"Metric '{name}' is not registered.")
        metric_fn = cls._registry[name]
        return metric_fn


@GenRecMetricFactory.register("hr")
def calc_metric_hr(
    topk_sids: Int[torch.Tensor, "B K C"],
    labels: Int[torch.Tensor, "B C"],
    train_dataset: GenRecDataset,
    **kwargs: Any,
) -> Dict[str, float]:
    """Calculate Hit Rate (HR) metric.

    Args:
        topk_sids (Int[torch.Tensor, "B K C"]): Top-K item sids predicted by the model
            sorted by their generation scores in descending order.
        labels (Int[torch.Tensor, "B C"]): Ground-truth item sids.
        train_dataset (GenRecDataset): Training dataset used to compute item-level statistics.

    Returns:
        Dict[str, float]: Mapping from "hr@K" to its computed value.
    """
    K = topk_sids.shape[1]
    hits = (topk_sids == labels.unsqueeze(1)).all(dim=-1).float()
    hr = hits.any(dim=1).float().mean().item()
    return {f"hr@{K}": hr}


@GenRecMetricFactory.register("ndcg")
def calc_metric_ndcg(
    topk_sids: Int[torch.Tensor, "B K C"],
    labels: Int[torch.Tensor, "B C"],
    train_dataset: GenRecDataset,
    **kwargs: Any,
) -> Dict[str, float]:
    """Calculate Normalized Discounted Cumulative Gain (NDCG) metric.

    Args:
        topk_sids (Int[torch.Tensor, "B K C"]): Top-K item sids predicted by the model
            sorted by their generation scores in descending order.
        labels (Int[torch.Tensor, "B C"]): Ground-truth item sids.
        train_dataset (GenRecDataset): Training dataset used to compute item-level statistics.

    Returns:
        Dict[str, float]: Mapping from "ndcg@K" to its computed value.
    """
    K = topk_sids.shape[1]
    hits = (topk_sids == labels.unsqueeze(1)).all(dim=-1).float()
    discounts = 1.0 / torch.log2(torch.arange(2, K + 2, device=topk_sids.device, dtype=torch.float32))
    dcg = (hits * discounts.view(1, -1)).max(dim=1).values
    ndcg = dcg.mean().item()
    return {f"ndcg@{K}": ndcg}


@GenRecMetricFactory.register("popularity")
def calc_metric_popularity(
    topk_sids: Int[torch.Tensor, "B K C"],
    labels: Int[torch.Tensor, "B C"],
    train_dataset: GenRecDataset,
    p: Sequence[float] = (0.1, 0.2),
    **kwargs: Any,
) -> Dict[str, float]:
    """Calculate Top-p Popularity metric.

    Args:
        topk_sids (Int[torch.Tensor, "B K C"]): Top-K item sids predicted by the model
            sorted by their generation scores in descending order.
        labels (Int[torch.Tensor, "B C"]): Ground-truth item sids.
        train_dataset (GenRecDataset): Training dataset used to derive popularity statistics.
        p (Sequence[float]): Popularity thresholds (between 0 and 1) defining popular items.

    Returns:
        Dict[str, float]: Dictionary mapping "popularity@K-p" to its float value.
    """
    device = topk_sids.device
    B, K, _ = topk_sids.shape

    sid2item = train_dataset.sid2item
    topk_item_ids: Int[torch.Tensor, "B K"] = batch_sid_to_item(topk_sids, sid2item)

    item_popularity = torch.as_tensor(train_dataset.item_popularity, device=device, dtype=torch.float32)
    I = item_popularity.shape[0] - 1
    pred_cnt = torch.bincount(topk_item_ids.reshape(-1), minlength=I + 1)
    pred_sum = float(B * K)
    popularity_rank = torch.argsort(item_popularity[1:], descending=True)

    results: Dict[str, float] = {}
    for threshold in p:
        popular_threshold = max(1, int(I * threshold))
        selected_items = popularity_rank[:popular_threshold]
        pop_sum = pred_cnt.index_select(0, selected_items + 1).sum().float()
        results[f"popularity@{K}-{threshold}"] = (pop_sum / pred_sum).item()

    return results


@GenRecMetricFactory.register("unpopularity")
def calc_metric_unpopularity(
    topk_sids: Int[torch.Tensor, "B K C"],
    labels: Int[torch.Tensor, "B C"],
    train_dataset: GenRecDataset,
    p: Sequence[float] = (0.1, 0.2),
    **kwargs: Any,
) -> Dict[str, float]:
    """Calculate Bottom-p Unpopularity metric.

    Args:
        topk_sids (Int[torch.Tensor, "B K C"]): Top-K item sids predicted by the model
            sorted by their generation scores in descending order.
        labels (Int[torch.Tensor, "B C"]): Ground-truth item sids.
        train_dataset (GenRecDataset): Training dataset used to derive popularity statistics.
        p (Sequence[float]): Unpopularity thresholds (between 0 and 1) defining unpopular items.

    Returns:
        Dict[str, float]: Dictionary mapping "unpopularity@K-p" to its float value.
    """
    device = topk_sids.device
    B, K, _ = topk_sids.shape

    sid2item = train_dataset.sid2item
    topk_item_ids: Int[torch.Tensor, "B K"] = batch_sid_to_item(topk_sids, sid2item)

    item_popularity = torch.as_tensor(train_dataset.item_popularity, device=device, dtype=torch.float32)
    I = item_popularity.shape[0] - 1
    pred_cnt = torch.bincount(topk_item_ids.reshape(-1), minlength=I + 1)
    pred_sum = float(B * K)
    unpopularity_rank = torch.argsort(item_popularity[1:], descending=False)

    results: Dict[str, float] = {}
    for threshold in p:
        unpopular_threshold = max(1, int(I * threshold))
        selected_items = unpopularity_rank[:unpopular_threshold]
        unpop_sum = pred_cnt.index_select(0, selected_items + 1).sum().float()
        results[f"unpopularity@{K}-{threshold}"] = (unpop_sum / pred_sum).item()

    return results
