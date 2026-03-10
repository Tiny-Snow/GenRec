"""Models for generative recommendation tasks."""

__all__ = []

from .base import (
    GenRecModel,
    GenRecModelConfig,
    GenRecModelConfigFactory,
    GenRecModelFactory,
    GenRecOutput,
    GenRecOutputFactory,
    ShiftRightMixin,
)

__all__ += [
    "GenRecModel",
    "GenRecModelFactory",
    "GenRecModelConfig",
    "GenRecModelConfigFactory",
    "GenRecOutput",
    "GenRecOutputFactory",
    "ShiftRightMixin",
]

from .tiger import TIGERModel, TIGERModelConfig, TIGERModelOutput

__all__ += [
    "TIGERModel",
    "TIGERModelConfig",
    "TIGERModelOutput",
]

from .tiger_legacy import TIGERLegacyModel, TIGERLegacyModelConfig, TIGERLegacyModelOutput

__all__ += [
    "TIGERLegacyModel",
    "TIGERLegacyModelConfig",
    "TIGERLegacyModelOutput",
]
