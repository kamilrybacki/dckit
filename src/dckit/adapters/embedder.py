"""Embedder protocol — bring-your-own embedding model."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Embedder(Protocol):
    """Embeds text into fixed-dimension dense vectors.

    Implementations may wrap sentence-transformers, fastembed, OpenAI
    embeddings, or any other model returning ndarray.

    Examples
    --------
    Sentence-transformers wrapper::

        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("BAAI/bge-small-en-v1.5")

        class STEmbedder:
            dim = 384
            def embed(self, texts):
                return model.encode(texts, normalize_embeddings=True)

    Callable wrapper — any callable matching ``(list[str]) -> ndarray`` is
    accepted by `discover()` and `OracleJudge` directly.
    """

    @property
    def dim(self) -> int: ...

    def embed(self, texts: list[str]) -> np.ndarray: ...
