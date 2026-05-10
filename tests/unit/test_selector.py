"""AreaMMR + FullMMR selectors."""

from __future__ import annotations

import numpy as np
import pytest

from dckit.adapters.vector_db import Candidate
from dckit.selector import AreaMMR, FullMMR


@pytest.mark.unit
def test_area_mmr_diverse_at_gamma_high(candidates: list[Candidate]) -> None:
    """High gamma forces diversity across areas."""
    chosen = AreaMMR(gamma=0.5).select(candidates, k=4)
    areas = [c.payload["codebook_idx"] for c in chosen]
    assert len(set(areas)) == 4, f"expected 4 distinct areas, got {areas}"


@pytest.mark.unit
def test_area_mmr_top_k_at_gamma_zero(candidates: list[Candidate]) -> None:
    """gamma=0 reduces to top-k by score."""
    chosen = AreaMMR(gamma=0.0).select(candidates, k=3)
    assert [c.id for c in chosen] == ["a", "b", "c"]


@pytest.mark.unit
def test_area_mmr_empty_returns_empty() -> None:
    assert AreaMMR().select([], k=5) == []


@pytest.mark.unit
def test_area_mmr_caps_k_at_candidate_count(candidates: list[Candidate]) -> None:
    chosen = AreaMMR(gamma=0.1).select(candidates, k=100)
    assert len(chosen) == len(candidates)


@pytest.mark.unit
def test_area_mmr_handles_missing_payload() -> None:
    cands = [
        Candidate(id="a", score=0.9, payload={}),
        Candidate(id="b", score=0.8, payload={}),
    ]
    chosen = AreaMMR(gamma=0.5).select(cands, k=2)
    assert len(chosen) == 2


@pytest.mark.unit
def test_area_mmr_custom_payload_key() -> None:
    cands = [
        Candidate(id="a", score=0.9, payload={"area": 0}),
        Candidate(id="b", score=0.85, payload={"area": 0}),
        Candidate(id="c", score=0.8, payload={"area": 1}),
    ]
    chosen = AreaMMR(gamma=0.5, payload_key="area").select(cands, k=2)
    assert {c.id for c in chosen} == {"a", "c"}


@pytest.mark.unit
def test_full_mmr_requires_vectors() -> None:
    cands = [Candidate(id="a", score=0.9, payload={"codebook_idx": 0})]
    with pytest.raises(ValueError, match="vector"):
        FullMMR().select(cands, query_vec=np.ones(4, dtype=np.float32), k=1)


@pytest.mark.unit
def test_full_mmr_picks_diverse_with_vectors() -> None:
    """With γ + low λ, MMR picks across both clusters even when relevance
    is concentrated in cluster 0."""
    rng = np.random.default_rng(0)
    centers = rng.standard_normal((2, 8)).astype(np.float32) * 3
    cands = []
    for i in range(6):
        c = i % 2
        v = centers[c] + rng.standard_normal(8).astype(np.float32) * 0.1
        v /= np.linalg.norm(v)
        cands.append(
            Candidate(
                id=f"p{i}",
                score=1.0 - i * 0.01,
                payload={"codebook_idx": c},
                vector=v,
            )
        )
    q = centers[0] / np.linalg.norm(centers[0])
    chosen = FullMMR(lambda_=0.3, gamma=0.5).select(cands, query_vec=q, k=2)
    areas = [c.payload["codebook_idx"] for c in chosen]
    assert len(set(areas)) == 2
