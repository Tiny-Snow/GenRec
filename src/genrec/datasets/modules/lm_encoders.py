"""Language-model-based encoders for item textual metadata (e.g., titles) embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Protocol, Sequence, Type, runtime_checkable

from jaxtyping import Float
import numpy as np
from sentence_transformers import SentenceTransformer

__all__ = [
    "LMEncoder",
    "LMEncoderFactory",
    "SentenceT5Encoder",
]


class LMEncoderFactory:  # pragma: no cover - factory class
    """Factory for creating `LMEncoder` instances."""

    _registry: dict[str, Type[LMEncoder]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Type[LMEncoder]], Type[LMEncoder]]:
        """Decorator to register a `LMEncoder` implementation."""

        def decorator(encoder_cls: Type[LMEncoder]) -> Type[LMEncoder]:
            if name in cls._registry:
                raise ValueError(f"LM encoder '{name}' is already registered.")
            cls._registry[name] = encoder_cls
            return encoder_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs) -> LMEncoder:
        """Creates an instance of a registered `LMEncoder`."""
        if name not in cls._registry:
            raise ValueError(f"LM encoder '{name}' is not registered.")
        encoder_cls = cls._registry[name]
        return encoder_cls(**kwargs)


@runtime_checkable
class LMEncoder(Protocol):  # pragma: no cover - protocol
    """Protocol describing minimal interface for text encoders."""

    @property
    def embedding_dim(self) -> int: ...

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Encodes a batch of input texts into dense embeddings.

        Args:
            texts (Sequence[str]): Sequence of input text strings.

        Returns:
            np.ndarray: Array of shape (len(texts), embedding_dim) containing the text embeddings.
        """
        ...


@LMEncoderFactory.register("sentence_t5")
class SentenceT5Encoder:
    """Utility wrapper around Sentence-T5 models from SentenceTransformers."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/sentence-t5-base",
        device: str | None = None,
        local_model_dir: str | None = None,
        allow_download: bool = True,
    ) -> None:
        """Initialises the encoder.

        Args:
            model_name (str): Hugging Face identifier for the model to load when a local directory
                is not provided.
            device (Optional[str]): Optional device string forwarded to SentenceTransformers.
            local_model_dir (Optional[str]): Directory containing a cached model or to be used as
                a cache location.
            allow_download (bool): Whether to download the model if `local_model_dir` does not
                exist yet.

        Raises:
            FileNotFoundError: If `local_model_dir` is supplied, missing, and
                downloads are disabled.
        """
        model_source = model_name

        self._model_dir: Path | None = None
        cache_folder: str | None = None
        if local_model_dir is not None:
            resolved_dir = Path(local_model_dir).expanduser().resolve()
            self._model_dir = resolved_dir
            if resolved_dir.exists():  # pragma: no cover - path exists
                model_source = str(resolved_dir)
            else:
                if not allow_download:
                    raise FileNotFoundError(f"Local model directory '{resolved_dir}' not found and downloads disabled.")
                resolved_dir.mkdir(parents=True, exist_ok=True)
                cache_folder = str(resolved_dir)

        if cache_folder is not None:
            self._model = SentenceTransformer(model_source, device=device, cache_folder=cache_folder)
        else:
            self._model = SentenceTransformer(model_source, device=device)

    @property
    def embedding_dim(self) -> int:
        """Returns the dimensionality of the sentence embeddings.

        Returns:
            Integer dimensionality reported by the underlying model.

        Raises:
            ValueError: If the transformer backend does not expose a dimension.
        """
        dimension = self._model.get_sentence_embedding_dimension()
        if dimension is None:  # pragma: no cover - defensive guard against optional typing
            raise ValueError("SentenceTransformer model did not report an embedding dimension.")
        return int(dimension)

    def encode(self, texts: Sequence[str]) -> Float[np.ndarray, "N D"]:
        """Embeds input texts into dense vectors.

        Args:
            texts (Sequence[str]): Sequence of strings to encode.

        Returns:
            Float[np.ndarray, "N D"]: Array of embedding vectors corresponding to `texts`.

        Raises:
            TypeError: If the underlying model returns an unexpected type.
        """
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        embeddings = self._model.encode(list(texts), batch_size=128, convert_to_numpy=True, show_progress_bar=True)
        return np.asarray(embeddings, dtype=np.float32)
