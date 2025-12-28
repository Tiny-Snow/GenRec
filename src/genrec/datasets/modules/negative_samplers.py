"""Negative sampling utilities for GenRec data pipelines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Tuple, Type

import numba
import numpy as np
from jaxtyping import Int

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..base import RecDataset

__all__ = [
    "NegativeSampler",
    "NegativeSamplerFactory",
    "PopularityNegativeSampler",
    "UniformNegativeSampler",
]


class NegativeSamplerFactory:  # pragma: no cover - factory class
    """Factory for creating `NegativeSampler` instances."""

    _registry: dict[str, Type[NegativeSampler]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Type[NegativeSampler]], Type[NegativeSampler]]:
        """Decorator to register a `NegativeSampler` implementation."""

        def decorator(sampler_cls: Type[NegativeSampler]) -> Type[NegativeSampler]:
            if name in cls._registry:
                raise ValueError(f"Negative sampler '{name}' is already registered.")
            cls._registry[name] = sampler_cls
            return sampler_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs) -> NegativeSampler:
        """Creates an instance of a registered `NegativeSampler`."""
        if name not in cls._registry:
            raise ValueError(f"Negative sampler '{name}' is not registered.")
        sampler_cls = cls._registry[name]
        return sampler_cls(**kwargs)


class NegativeSampler(ABC):
    """Abstract base class for callable negative samplers."""

    def __init__(self, dataset: RecDataset) -> None:
        """Initializes the negative sampler.

        Args:
            dataset (RecDataset): Dataset from which to sample negatives.
        """
        self._item_size = dataset.item_size

    def __call__(
        self,
        history: Int[np.ndarray, "B L"],
        num_samples: int,
        batch_seed: int | None = None,
    ) -> Int[np.ndarray, "B num_samples"]:
        """Generates negative samples for given users. Negative sampler samples items
        from the interval `[1, item_size]` and excludes items in the user history.

        Args:
            history (Int[np.ndarray, "B L"]): Batch of user interaction histories with shape (B, L).
            num_samples (int): Number of negative samples to generate per user.
            batch_seed (Optional[int]): Random seed for the entire batch to ensure reproducibility
                across multi-process data loading.

        Returns:
            Int[np.ndarray, "B num_samples"]: Array containing the sampled negative item IDs.
        """
        B = history.shape[0]
        if batch_seed is None:  # pragma: no cover - generate random seed
            base_seed = np.random.randint(0, 2**31 - 1)
        else:
            base_seed = batch_seed
        rng = np.random.default_rng(base_seed)
        user_seeds = rng.integers(0, 2**31 - 1, size=B, dtype=np.int32)
        negatives = self._sample(history, num_samples, user_seeds)
        return negatives

    @abstractmethod
    def _sample(  # pragma: no cover - abstract method
        self,
        history: Int[np.ndarray, "B L"],
        num_samples: int,
        seeds: Int[np.ndarray, "B"],
    ) -> Int[np.ndarray, "B num_samples"]:
        """Generates negative samples for given users.

        Args:
            history (Int[np.ndarray, "B L"]): Batch of user interaction histories with shape (B, L).
            num_samples (int): Number of negative samples to generate per user.
            seeds (Int[np.ndarray, "B"]): Array containing random seeds for each user.

        Returns:
            Int[np.ndarray, "B num_samples"]: Array containing the sampled negative item IDs.
        """
        ...


_LCG_MULTIPLIER = np.uint64(6364136223846793005)
_LCG_INCREMENT = np.uint64(1442695040888963407)
_LCG_MASK = np.uint64(0xFFFFFFFFFFFFFFFF)


@numba.njit(inline="always")
def _lcg_next(  # pragma: no cover - numba-jitted code
    state: np.uint64,
) -> np.uint64:
    return (state * _LCG_MULTIPLIER + _LCG_INCREMENT) & _LCG_MASK


@numba.njit(inline="always")
def _lcg_randint(  # pragma: no cover - numba-jitted code
    state: np.uint64,
    lower: int,
    upper: int,
) -> Tuple[np.uint64, int]:
    span = upper - lower + 1
    if span <= 0:
        raise ValueError("Upper bound must be greater than or equal to lower bound.")
    state = _lcg_next(state)
    value = lower + int(state % np.uint64(span))
    return state, value


@numba.njit
def _uniform_sample_numba(  # pragma: no cover - numba-jitted code
    lower: int,
    upper: int,
    history: Int[np.ndarray, "B L"],
    num_samples: int,
    seeds: Int[np.ndarray, "B"],
) -> Int[np.ndarray, "B num_samples"]:
    """Draws uniform negative samples while excluding positives using Numba.

    Args:
        lower (int): Inclusive lower bound of the item identifier range.
        upper (int): Inclusive upper bound of the item identifier range.
        history (Int[np.ndarray, "B L"]): Array containing user interaction histories.
        num_samples (int): Number of negative samples to draw per user.
        seeds (Int[np.ndarray, "B"]): Array containing random seeds for each user.

    Returns:
        Int[np.ndarray, "B num_samples"]: Array of sampled negative item IDs.
    """
    B, L = history.shape
    negatives = np.empty((B, num_samples), dtype=np.int32)
    for i in numba.prange(B):
        user_history = history[i]
        state = np.uint64(np.uint32(seeds[i])) | np.uint64(1)
        count = 0
        while count < num_samples:
            state, item = _lcg_randint(state, lower, upper)
            in_history = False
            for j in range(L):
                if user_history[j] == item:
                    in_history = True
                    break
            if not in_history:
                negatives[i, count] = item
                count += 1
    return negatives


@NegativeSamplerFactory.register("uniform")
class UniformNegativeSampler(NegativeSampler):
    """Uniformly samples negatives within a continuous item-ID interval."""

    def __init__(self, dataset: RecDataset) -> None:
        """Initializes the uniform negative sampler.

        Args:
            dataset (RecDataset): Dataset from which to sample negatives.
        """
        super().__init__(dataset)

    def _sample(
        self,
        history: Int[np.ndarray, "B L"],
        num_samples: int,
        seeds: Int[np.ndarray, "B"],
    ) -> Int[np.ndarray, "B num_samples"]:
        """Generates negative samples uniformly for given users."""
        return _uniform_sample_numba(1, self._item_size, history, num_samples, seeds)


@numba.njit
def _binary_search(  # pragma: no cover - numba-jitted code
    cdf: Int[np.ndarray, "I+1"],
    value: int,
) -> int:
    """Performs binary search to find the index of the smallest element in `cdf`
    that is greater than or equal to `value`.

    Args:
        cdf (Int[np.ndarray, "I+1"]): 1D array of cumulative distribution function values (must be sorted).
        value (int): Value to search for in the CDF.

    Returns:
        int: Index of the smallest element in `cdf` that is >= `value`.
    """
    low = 0
    high = cdf.shape[0] - 1
    while low < high:
        mid = (low + high) // 2
        if cdf[mid] < value:
            low = mid + 1
        else:
            high = mid
    return low


@numba.njit
def _popularity_sample_numba(  # pragma: no cover - numba-jitted code
    cdf: Int[np.ndarray, "I+1"],
    history: Int[np.ndarray, "B L"],
    num_samples: int,
    seeds: Int[np.ndarray, "B"],
) -> Int[np.ndarray, "B num_samples"]:
    """Draws popularity-based negative samples while excluding positives using Numba.

    Args:
        cdf (Int[np.ndarray, "I+1"]): 1D cumulative distribution describing item popularity, with
            index 0 reserved for padding.
        history (Int[np.ndarray, "B L"]): Array containing user interaction histories.
        num_samples (int): Number of negative samples to draw per user.
        seeds (Int[np.ndarray, "B"]): Array containing random seeds for each user.

    Returns:
        Int[np.ndarray, "B num_samples"]: Array of sampled negative item IDs.
    """
    B, L = history.shape
    max_cdf_value = cdf[-1]
    negatives = np.empty((B, num_samples), dtype=np.int32)
    for i in numba.prange(B):
        user_history = history[i]
        state = np.uint64(np.uint32(seeds[i])) | np.uint64(1)
        count = 0
        while count < num_samples:
            state = _lcg_next(state)
            rand_value = int(state % np.uint64(max_cdf_value)) + 1
            item = _binary_search(cdf, rand_value)
            in_history = False
            for j in range(L):
                if user_history[j] == item:
                    in_history = True
                    break
            if not in_history:
                negatives[i, count] = item
                count += 1
    return negatives


@NegativeSamplerFactory.register("popularity")
class PopularityNegativeSampler(NegativeSampler):
    """Samples negatives based on item popularity within a continuous item-ID interval."""

    def __init__(self, dataset: RecDataset) -> None:
        """Initializes the popularity-based negative sampler.

        Args:
            dataset (RecDataset): Dataset from which to sample negatives.
        """
        super().__init__(dataset)

        popularity: Int[np.ndarray, "I+1"] = dataset.train_item_popularity
        self.cdf = np.cumsum(popularity, dtype=np.int64)

    def _sample(
        self,
        history: Int[np.ndarray, "B L"],
        num_samples: int,
        seeds: Int[np.ndarray, "B"],
    ) -> Int[np.ndarray, "B num_samples"]:
        """Generates negative samples based on item popularity for given users."""
        return _popularity_sample_numba(self.cdf, history, num_samples, seeds)
