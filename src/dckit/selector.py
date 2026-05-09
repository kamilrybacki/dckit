"""Diversity-aware context selectors.

`AreaMMR` is the production-validated selector: O(N·K) per query, no inter-
candidate cosine, no vector fetch. `FullMMR` is the standard MMR with both
inter-candidate redundancy and codebook-area penalty.

Empirical finding from the dckit benchmark: AreaMMR is Pareto-optimal for
diverse-context tasks at γ ∈ [0.10, 0.20]. FullMMR adds compute without
recall lift on the 56-vault corpus.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


class HasScoreAndPayload(Protocol):
    """Minimal candidate shape: cosine score + payload dict.

    Compatible with qdrant ScoredPoint, weaviate result rows, custom dicts
    wrapped in a SimpleNamespace, etc.
    """

    score: float
    payload: dict[str, Any] | None


@dataclass(frozen=True)
class AreaMMR:
    """Area-only MMR — categorical penalty, no inter-candidate cosine.

    Score formula:
        score(d) = lambda_ * sim(q, d) − gamma * [area(d) ∈ selected_areas]

    Parameters
    ----------
    lambda_:
        Weight on the relevance term. Typical: 0.7. Has marginal effect when
        γ > 0; relevance is already pre-baked into ``candidate.score``.
    gamma:
        Categorical-redundancy penalty. 0 disables diversity (returns top-k by
        score). 0.10 — light; 0.20 — moderate; 0.40 — strong (sometimes hurts
        recall).
    payload_key:
        Field on candidate.payload that holds the discrete area id. Defaults
        to ``l1_argmax`` (matching the dckit ingest convention).
    """

    lambda_: float = 0.7
    gamma: float = 0.10
    payload_key: str = "l1_argmax"

    def select(self, candidates: list[Any], k: int) -> list[Any]:
        if not candidates:
            return []
        k = min(k, len(candidates))
        selected: list[Any] = []
        selected_areas: set[int] = set()
        remaining = list(candidates)

        for _ in range(k):
            best, best_score = None, -np.inf
            for c in remaining:
                area = self._area_of(c)
                penalty = (
                    -self.gamma
                    if (self.gamma > 0 and area is not None and area in selected_areas)
                    else 0.0
                )
                relevance = self.lambda_ * float(getattr(c, "score", 0.0))
                total = relevance + penalty
                if total > best_score:
                    best_score, best = total, c
            if best is None:
                break
            selected.append(best)
            area = self._area_of(best)
            if area is not None:
                selected_areas.add(area)
            remaining.remove(best)

        return selected

    def _area_of(self, c: Any) -> int | None:
        payload = getattr(c, "payload", None) or {}
        v = payload.get(self.payload_key)
        return int(v) if v is not None else None


@dataclass(frozen=True)
class FullMMR:
    """Standard MMR with codebook-area diversity penalty.

    Requires candidate vectors. Use only when AreaMMR is insufficient.
    """

    lambda_: float = 0.7
    gamma: float = 0.10
    payload_key: str = "l1_argmax"

    def select(
        self,
        candidates: list[Any],
        query_vec: np.ndarray,
        k: int,
    ) -> list[Any]:
        n = len(candidates)
        if n == 0:
            return []
        k = min(k, n)

        cand_vecs = self._stack_vectors(candidates)
        q = query_vec.astype(np.float32)
        q /= np.linalg.norm(q) + 1e-10

        cand_unit = cand_vecs / (np.linalg.norm(cand_vecs, axis=1, keepdims=True) + 1e-10)
        q_sims = cand_unit @ q
        sim_mat = cand_unit @ cand_unit.T

        selected: list[int] = []
        selected_areas: set[int] = set()
        remaining = list(range(n))

        for _ in range(k):
            best_i, best_score = -1, -np.inf
            for i in remaining:
                rel = self.lambda_ * float(q_sims[i])
                red = (1 - self.lambda_) * float(np.max(sim_mat[i, selected])) if selected else 0.0
                area = self._area_of(candidates[i])
                penalty = (
                    -self.gamma
                    if (self.gamma > 0 and area is not None and area in selected_areas)
                    else 0.0
                )
                score = rel - red + penalty
                if score > best_score:
                    best_score, best_i = score, i
            if best_i == -1:
                break
            selected.append(best_i)
            area = self._area_of(candidates[best_i])
            if area is not None:
                selected_areas.add(area)
            remaining.remove(best_i)

        return [candidates[i] for i in selected]

    def _area_of(self, c: Any) -> int | None:
        payload = getattr(c, "payload", None) or {}
        v = payload.get(self.payload_key)
        return int(v) if v is not None else None

    @staticmethod
    def _stack_vectors(candidates: list[Any]) -> np.ndarray:
        vecs = []
        for c in candidates:
            v = getattr(c, "vector", None)
            if v is None:
                raise ValueError("FullMMR requires candidate.vector; use AreaMMR otherwise")
            vecs.append(v if isinstance(v, np.ndarray) else np.asarray(v, dtype=np.float32))
        return np.stack(vecs).astype(np.float32)
