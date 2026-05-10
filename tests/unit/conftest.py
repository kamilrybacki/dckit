"""Shared fixtures for unit tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from dckit.adapters.vector_db import Candidate
from dckit.codebook import Codebook


@pytest.fixture()
def small_codebook() -> Codebook:
    rng = np.random.default_rng(0)
    embeddings = rng.standard_normal((4, 8)).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    return Codebook(
        embeddings=embeddings,
        labels=("00 — Alpha", "10 — Beta", "20 — Gamma", "30 — Delta"),
        score_min=-1.0,
        score_max=1.0,
        metadata={"test": True},
    )


@pytest.fixture()
def candidates() -> list[Candidate]:
    return [
        Candidate(id="a", score=0.95, payload={"codebook_idx": 0}),
        Candidate(id="b", score=0.92, payload={"codebook_idx": 0}),
        Candidate(id="c", score=0.91, payload={"codebook_idx": 1}),
        Candidate(id="d", score=0.85, payload={"codebook_idx": 2}),
        Candidate(id="e", score=0.80, payload={"codebook_idx": 3}),
    ]


class FakeBackend:
    """In-memory VectorBackend impl for testing."""

    def __init__(self, points: list[Candidate]) -> None:
        self._points = points

    def query(
        self,
        vector: np.ndarray,
        limit: int,
        *,
        with_payload: bool | list[str] = True,
        with_vectors: bool = False,
        filter: dict[str, Any] | None = None,
    ) -> list[Candidate]:
        if any(p.vector is None for p in self._points):
            return self._points[:limit]
        scores = []
        for p in self._points:
            assert p.vector is not None
            v = p.vector / (np.linalg.norm(p.vector) + 1e-10)
            q = vector / (np.linalg.norm(vector) + 1e-10)
            scores.append((float(v @ q), p))
        scores.sort(key=lambda x: -x[0])
        return [
            Candidate(
                id=p.id,
                score=s,
                payload=p.payload,
                vector=p.vector if with_vectors else None,
            )
            for s, p in scores[:limit]
        ]

    def scroll(
        self,
        limit: int,
        *,
        offset: Any | None = None,
        with_payload: bool | list[str] = True,
        with_vectors: bool = False,
    ) -> tuple[list[Candidate], Any | None]:
        start = int(offset or 0)
        end = min(start + limit, len(self._points))
        out = [
            Candidate(
                id=p.id,
                score=p.score,
                payload=p.payload,
                vector=p.vector if with_vectors else None,
            )
            for p in self._points[start:end]
        ]
        next_offset: int | None = end if end < len(self._points) else None
        return out, next_offset

    def set_payload(self, ids: list[str], payload: dict[str, Any]) -> None:
        wanted = set(ids)
        for p in self._points:
            if p.id in wanted:
                p.payload.update(payload)


@pytest.fixture()
def fake_backend() -> Any:
    return FakeBackend


def make_synthetic_corpus(
    n: int = 200, dim: int = 16, k_clusters: int = 4, seed: int = 0
) -> list[Candidate]:
    """Build a corpus with k well-separated Gaussian clusters."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((k_clusters, dim)).astype(np.float32) * 3.0
    points: list[Candidate] = []
    for i in range(n):
        c = i % k_clusters
        v = centers[c] + rng.standard_normal(dim).astype(np.float32) * 0.2
        v /= np.linalg.norm(v) + 1e-10
        points.append(
            Candidate(
                id=f"p{i}",
                score=0.0,
                payload={"text": f"cluster-{c} sample {i} keywords domain-{c}"},
                vector=v,
            )
        )
    return points


@pytest.fixture()
def synthetic_corpus() -> list[Candidate]:
    return make_synthetic_corpus()


class StubLLM:
    """LLMClient stub returning canned responses keyed by prompt fragments."""

    def __init__(self, responses: dict[str, str]) -> None:
        self._responses = responses
        self.calls: list[tuple[str, str]] = []

    def chat(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        system: str | None = None,
    ) -> str:
        self.calls.append((model, prompt))
        for key, val in self._responses.items():
            if key in prompt:
                return val
        return ""


@pytest.fixture()
def stub_llm_factory() -> Any:
    return StubLLM


def candidate_with(score: float, area: int) -> Candidate:
    return Candidate(id=f"x{area}-{score}", score=score, payload={"codebook_idx": area})


def make_simple_namespace_candidate(score: float, area: int) -> SimpleNamespace:
    return SimpleNamespace(score=score, payload={"codebook_idx": area}, id="ns")
