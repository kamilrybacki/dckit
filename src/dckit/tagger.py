"""Class-mean projection tagger.

Maps embedding vectors to discrete codebook areas via T·v matmul.
One pass per ingest batch; ~1 ms per 1k vectors on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .codebook import Codebook


@dataclass(frozen=True)
class TagResult:
    """Per-vector classification result.

    Attributes
    ----------
    argmax: int   ── selected area index in [0, K).
    label:  str   ── human-readable label from codebook.
    scores: ndarray  ── (K,) cosine projection scores against each class mean.
    confidence: float  ── normalised score for the selected class in [0, 1].
    """

    argmax: int
    label: str
    scores: np.ndarray
    confidence: float


class Tagger:
    """Wraps a Codebook with batched tagging logic."""

    def __init__(self, codebook: Codebook) -> None:
        self._cb = codebook
        self._range = codebook.max_val - codebook.min_val

    def project(self, vectors: np.ndarray) -> np.ndarray:
        """Compute (N, K) raw cosine scores against class means."""
        if vectors.ndim != 2:
            raise ValueError(f"vectors must be 2-D, got shape {vectors.shape}")
        if vectors.shape[1] != self._cb.dim:
            raise ValueError(
                f"vector dim {vectors.shape[1]} != codebook dim {self._cb.dim}"
            )
        return vectors.astype(np.float32) @ self._cb.embeddings.T

    def project_normalised(self, vectors: np.ndarray) -> np.ndarray:
        """Project + min-max normalise to [0, 1] using calibration constants."""
        scores = self.project(vectors)
        if self._range > 1e-10:
            return np.clip((scores - self._cb.min_val) / self._range, 0.0, 1.0)
        return np.zeros_like(scores)

    def tag(self, vectors: np.ndarray) -> np.ndarray:
        """Return argmax area indices as (N,) int64 array — fastest path."""
        scores = self.project(vectors)
        return scores.argmax(axis=1).astype(np.int64)

    def tag_detailed(self, vectors: np.ndarray) -> list[TagResult]:
        """Return TagResult per vector (slower; use for diagnostics / single calls)."""
        norm = self.project_normalised(vectors)
        results: list[TagResult] = []
        for row in norm:
            argmax = int(row.argmax())
            results.append(
                TagResult(
                    argmax=argmax,
                    label=self._cb.label(argmax),
                    scores=row,
                    confidence=float(row[argmax]),
                )
            )
        return results

    def labels_for(self, indices: np.ndarray) -> list[str]:
        """Convert (N,) int array to list of labels."""
        return [self._cb.label(int(i)) for i in indices]
