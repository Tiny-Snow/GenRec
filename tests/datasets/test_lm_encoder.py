from __future__ import annotations

from pathlib import Path
from typing import Any, List, cast

import numpy as np
import pytest

from genrec.datasets.modules import lm_encoders as lm_mod
from genrec.datasets.modules.lm_encoders import SentenceT5Encoder


class _DummySentenceTransformer:
    instances: list["_DummySentenceTransformer"] = []

    def __init__(
        self,
        model_source: str,
        device: str | None = None,
        cache_folder: str | None = None,
    ) -> None:
        self.model_source = model_source
        self.device = device
        self.cache_folder = cache_folder
        self._dimension = 768
        self.encode_calls: list[list[str]] = []
        _DummySentenceTransformer.instances.append(self)

    def get_sentence_embedding_dimension(self) -> int:
        return self._dimension

    def encode(
        self,
        texts: List[str],
        batch_size: int = 128,
        convert_to_numpy: bool = True,
        show_progress_bar: bool = True,
    ) -> np.ndarray:
        assert convert_to_numpy
        assert batch_size == 128
        assert show_progress_bar is True
        self.encode_calls.append(list(texts))
        if not texts:
            return np.empty((0, self._dimension), dtype=np.float32)
        base = np.arange(len(texts) * self._dimension, dtype=np.float32)
        return base.reshape(len(texts), self._dimension)


def _install_dummy_encoder(monkeypatch: pytest.MonkeyPatch) -> None:
    _DummySentenceTransformer.instances.clear()
    monkeypatch.setattr(lm_mod, "SentenceTransformer", _DummySentenceTransformer)


def test_sentence_t5_encoder_encode_returns_float32(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_dummy_encoder(monkeypatch)

    encoder_cls = cast(Any, SentenceT5Encoder)
    encoder = encoder_cls(device="cpu")
    vectors = encoder.encode(["hello", "world"])

    assert vectors.shape == (2, 768)
    assert vectors.dtype == np.float32

    assert len(_DummySentenceTransformer.instances) == 1
    instance = _DummySentenceTransformer.instances[0]
    assert instance.device == "cpu"
    assert instance.cache_folder is None
    assert instance.encode_calls == [["hello", "world"]]


def test_sentence_t5_encoder_with_custom_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install_dummy_encoder(monkeypatch)

    cache_dir = tmp_path / "cache"
    assert not cache_dir.exists()

    encoder_cls = cast(Any, SentenceT5Encoder)
    encoder = encoder_cls(local_model_dir=str(cache_dir), allow_download=True)
    assert cache_dir.exists()

    assert len(_DummySentenceTransformer.instances) == 1
    instance = _DummySentenceTransformer.instances[0]
    assert instance.cache_folder == str(cache_dir)

    vectors = encoder.encode([])
    assert vectors.shape == (0, 768)


def test_sentence_t5_encoder_download_disabled_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install_dummy_encoder(monkeypatch)

    missing_dir = tmp_path / "missing"
    encoder_cls = cast(Any, SentenceT5Encoder)
    with pytest.raises(FileNotFoundError):
        encoder_cls(local_model_dir=str(missing_dir), allow_download=False)

    assert _DummySentenceTransformer.instances == []
