"""Utilities for data processing."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch

__all__ = [
    "SeedWorkerMixin",
    "numpy_to_torch",
    "pad_batch",
    "stack_batch",
]


def pad_batch(
    batch: List[Dict[str, np.ndarray]],
    direction: str = "right",
    pad_values: Optional[Dict[str, Any]] = None,
    max_length: Optional[int] = None,
    pad_to_multiple_of: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Generic batch padding utility.

    Args:
        batch (List[Dict[str, np.ndarray]]): List of samples, each a dict like
            {"field1": [...], "field2": [...]} where the first dimension corresponds
            to sequence length, and padding occurs along that dimension.
        direction (str): Either "left" or "right" padding direction.
        pad_values (Optional[Dict[str, Any]]): Padding values per field, e.g.,
            {"field1": 0, "field2": -100}. Defaults to 0 when unspecified.
        max_length (Optional[int]): Fixed length to pad to. If None, uses max length in batch.
            Note that if max_length is less than the longest sequence in the batch,
            all sequences will be padded to the longest sequence instead.
        pad_to_multiple_of (Optional[int]): If set, pads lengths to be multiples of this value.

    Returns:
        Dict[str, np.ndarray]: Dictionary mapping field names to padded batch data.
    """
    assert direction in ("left", "right"), "direction must be 'left' or 'right'."

    all_keys = batch[0].keys()
    pad_values = pad_values or {}

    if len(all_keys) == 0:
        return {}

    batch_max_length = max(sample[key].shape[0] for sample in batch for key in all_keys)
    if max_length is not None:
        batch_max_length = max(batch_max_length, max_length)

    if pad_to_multiple_of is not None and pad_to_multiple_of > 0:
        if batch_max_length % pad_to_multiple_of != 0:
            batch_max_length = ((batch_max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

    padded_batch: Dict[str, np.ndarray] = {}
    for key in all_keys:
        pad_value = pad_values.get(key, 0)
        field_rows: List[np.ndarray] = []
        for sample in batch:
            array = sample[key]
            seq_len = array.shape[0]
            pad_length = batch_max_length - seq_len
            if direction == "left":
                pad_width = (pad_length, 0)
            else:
                pad_width = (0, pad_length)
            padded_array = np.pad(
                array,
                (pad_width,) + ((0, 0),) * (array.ndim - 1),
                constant_values=pad_value,
            )
            field_rows.append(padded_array)
        padded_batch[key] = np.stack(field_rows, axis=0)

    return padded_batch


def stack_batch(batch: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """Generic batch stacking utility without padding.

    Args:
        batch (List[Dict[str, np.ndarray]]): List of samples, each a dict like {"field1": [...], ...}.

    Returns:
        Dict[str, np.ndarray]: Dictionary mapping field names to stacked batch data.
    """
    all_keys = batch[0].keys()
    stacked_batch: Dict[str, np.ndarray] = {}
    for key in all_keys:
        stacked_batch[key] = np.stack([sample[key] for sample in batch], axis=0)
    return stacked_batch


def numpy_to_torch(batch: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    """Converts a batch of numpy arrays to PyTorch tensors.

    Args:
        batch (Dict[str, np.ndarray]): Dictionary mapping field names to numpy arrays.

    Returns:
        Dict[str, torch.Tensor]: Dictionary mapping field names to PyTorch tensors.
    """
    torch_batch: Dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        torch_batch[key] = torch.from_numpy(value)
    return torch_batch


class SeedWorkerMixin:
    """Mixin class to provide per-worker random seeds in PyTorch data loaders.
    This is useful to ensure reproducibility when using multiple workers.
    """

    def __init__(self, global_seed: int = 42) -> None:
        """Initializes the seed worker mixin."""
        self._global_seed = global_seed
        self._rng = None
        self._worker_seed = None
        self._batch_cnt = 0

    def _init_rng_if_needed(self) -> None:
        """Initializes the random number generator for the current worker."""
        if self._rng is None:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is None:
                seed = self._global_seed
            else:  # pragma: no cover - multi-worker loading
                seed = self._global_seed + worker_info.seed
            self._worker_seed = seed
            self._rng = np.random.default_rng(seed)

    def next_batch_seed(self) -> int:
        """Generates a random seed for the current batch."""
        self._init_rng_if_needed()
        assert self._rng is not None
        assert self._worker_seed is not None

        seed = np.random.SeedSequence([self._worker_seed, self._batch_cnt]).generate_state(1)[0]
        self._batch_cnt += 1

        return int(seed)
