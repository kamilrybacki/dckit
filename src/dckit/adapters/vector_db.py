"""Vector-DB adapter protocols + reference Qdrant implementation.

The library is vector-DB-agnostic via the `VectorBackend` Protocol. Reference
impl: `QdrantBackend`. To use Milvus / Weaviate / pgvector / Chroma, implement
the Protocol; ~50 lines per backend.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

from ..exceptions import AdapterError

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass(frozen=True)
class Candidate:
    """Vector-DB-agnostic search result.

    Adapters convert their native row type into Candidate before returning.
    Selectors and the oracle consume Candidate, never the native types.
    """

    id: str
    score: float
    payload: dict[str, Any] = field(default_factory=dict)
    vector: np.ndarray | None = None


@runtime_checkable
class Closeable(Protocol):
    """Optional protocol for adapters that hold connections."""

    def close(self) -> None: ...


@runtime_checkable
class VectorBackend(Protocol):
    """Minimum surface required by dckit.

    Implementations: see `QdrantBackend` below for a canonical one.
    """

    def query(
        self,
        vector: np.ndarray,
        limit: int,
        *,
        with_payload: bool | list[str] = True,
        with_vectors: bool = False,
        filter: dict[str, Any] | None = None,
    ) -> list[Candidate]: ...

    def scroll(
        self,
        limit: int,
        *,
        offset: Any | None = None,
        with_payload: bool | list[str] = True,
        with_vectors: bool = False,
    ) -> tuple[list[Candidate], Any | None]: ...

    def set_payload(self, ids: list[str], payload: dict[str, Any]) -> None: ...


def iter_all(
    backend: VectorBackend,
    *,
    batch: int = 1000,
    with_vectors: bool = False,
    with_payload: bool | list[str] = True,
) -> Iterator[Candidate]:
    """Iterate every point in a backend via repeated scroll calls."""
    offset: Any | None = None
    while True:
        points, offset = backend.scroll(
            batch,
            offset=offset,
            with_payload=with_payload,
            with_vectors=with_vectors,
        )
        if not points:
            return
        yield from points
        if offset is None:
            return


class QdrantBackend:
    """Reference Qdrant implementation of `VectorBackend`.

    Lazily imports `qdrant_client` to keep it an optional dependency. Install
    via `pip install dckit[qdrant]`.
    """

    def __init__(
        self,
        collection: str,
        *,
        url: str = "http://localhost:6333",
        api_key: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise AdapterError(
                "qdrant-client not installed; pip install dckit[qdrant]"
            ) from exc
        self._collection = collection
        self._client = QdrantClient(url=url, api_key=api_key, timeout=timeout)

    @property
    def collection(self) -> str:
        return self._collection

    def query(
        self,
        vector: np.ndarray,
        limit: int,
        *,
        with_payload: bool | list[str] = True,
        with_vectors: bool = False,
        filter: dict[str, Any] | None = None,
    ) -> list[Candidate]:
        points = self._client.search(
            collection_name=self._collection,
            query_vector=vector.astype(np.float32).tolist(),
            limit=limit,
            with_payload=with_payload,
            with_vectors=with_vectors,
            query_filter=filter,
        )
        return [self._to_candidate(p) for p in points]

    def scroll(
        self,
        limit: int,
        *,
        offset: Any | None = None,
        with_payload: bool | list[str] = True,
        with_vectors: bool = False,
    ) -> tuple[list[Candidate], Any | None]:
        points, next_offset = self._client.scroll(
            collection_name=self._collection,
            limit=limit,
            offset=offset,
            with_payload=with_payload,
            with_vectors=with_vectors,
        )
        return [self._to_candidate(p, default_score=0.0) for p in points], next_offset

    def set_payload(self, ids: list[str], payload: dict[str, Any]) -> None:
        self._client.set_payload(
            collection_name=self._collection,
            payload=payload,
            points=ids,
        )

    def close(self) -> None:
        with contextlib.suppress(AttributeError, RuntimeError):
            self._client.close()

    @staticmethod
    def _to_candidate(p: Any, default_score: float = 0.0) -> Candidate:
        vec = getattr(p, "vector", None)
        if vec is not None and not isinstance(vec, np.ndarray):
            vec = np.asarray(vec, dtype=np.float32)
        return Candidate(
            id=str(p.id),
            score=float(getattr(p, "score", default_score)),
            payload=dict(p.payload) if getattr(p, "payload", None) else {},
            vector=vec,
        )
