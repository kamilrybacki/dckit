"""Discrete codebook: K class-mean vectors + labels.

A Codebook is the artefact produced by `discover` (or hand-authored) and
consumed by `tagger.Tagger` and `selector.AreaMMR`. It is intentionally
small, deterministic, and serialisable to a single .npz file.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Codebook:
    """Frozen K-class codebook over a fixed embedding dimension.

    Attributes
    ----------
    embeddings:
        Shape (K, D) class-mean vectors in the embedder's vector space.
        L2-normalised at construction.
    labels:
        Length-K list of human-readable area labels (e.g. "70 — Systems").
    score_min, score_max:
        Calibration constants used by tagger to map raw cosine scores to
        [0, 1]. Stored from discovery time; may be ignored by callers that
        do their own normalisation.
    metadata:
        Free-form provenance: model used, seed, corpus hash, timestamp.
    """

    embeddings: np.ndarray
    labels: tuple[str, ...]
    score_min: float
    score_max: float
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        if self.embeddings.ndim != 2:
            raise ValueError(f"embeddings must be 2-D, got shape {self.embeddings.shape}")
        if len(self.labels) != self.embeddings.shape[0]:
            raise ValueError(
                f"labels length {len(self.labels)} != K {self.embeddings.shape[0]}"
            )
        if self.score_max <= self.score_min:
            raise ValueError(f"score_max {self.score_max} <= score_min {self.score_min}")

    @property
    def k(self) -> int:
        return self.embeddings.shape[0]

    @property
    def dim(self) -> int:
        return self.embeddings.shape[1]

    def label(self, idx: int) -> str:
        return self.labels[idx]

    def content_hash(self) -> str:
        """Stable hash of (embeddings, labels). Provenance independent."""
        h = hashlib.sha256()
        h.update(self.embeddings.tobytes())
        h.update(("\x00".join(self.labels)).encode("utf-8"))
        return h.hexdigest()[:16]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            codebook_centers=self.embeddings.astype(np.float32),
            labels=np.array(self.labels, dtype=object),
            score_min=np.array([self.score_min], dtype=np.float32),
            score_max=np.array([self.score_max], dtype=np.float32),
        )
        sidecar = path.with_suffix(".json")
        sidecar.write_text(
            json.dumps(
                {
                    "k": self.k,
                    "dim": self.dim,
                    "labels": list(self.labels),
                    "score_min": self.score_min,
                    "score_max": self.score_max,
                    "content_hash": self.content_hash(),
                    "metadata": self.metadata,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> Codebook:
        path = Path(path)
        data = np.load(path, allow_pickle=True)
        # Disk key is `codebook_centers`; in-memory attribute keeps name
        # `embeddings` for clarity within the dataclass.
        embeddings = np.asarray(data["codebook_centers"], dtype=np.float32)
        labels = tuple(str(s) for s in data["labels"].tolist())
        score_min = float(data["score_min"][0])
        score_max = float(data["score_max"][0])
        sidecar = path.with_suffix(".json")
        metadata: dict[str, Any] = {}
        if sidecar.exists():
            try:
                metadata = json.loads(sidecar.read_text(encoding="utf-8")).get("metadata", {})
            except json.JSONDecodeError:
                metadata = {}
        return cls(
            embeddings=embeddings,
            labels=labels,
            score_min=score_min,
            score_max=score_max,
            metadata=metadata,
        )

    @classmethod
    def from_class_means(
        cls,
        vectors: np.ndarray,
        assignments: np.ndarray,
        labels: list[str],
        metadata: dict[str, Any] | None = None,
    ) -> Codebook:
        """Build codebook from per-cluster mean vectors.

        Parameters
        ----------
        vectors:
            (N, D) input vectors.
        assignments:
            (N,) integer cluster ids in range [0, K).
        labels:
            Length-K labels.
        """
        k = len(labels)
        if vectors.ndim != 2:
            raise ValueError("vectors must be 2-D")
        if assignments.shape[0] != vectors.shape[0]:
            raise ValueError("assignments length mismatch")
        means = np.zeros((k, vectors.shape[1]), dtype=np.float32)
        for i in range(k):
            mask = assignments == i
            if not mask.any():
                raise ValueError(f"empty cluster {i}; refine discovery before fitting")
            means[i] = vectors[mask].mean(axis=0)
        norms = np.linalg.norm(means, axis=1, keepdims=True) + 1e-10
        means /= norms

        scores = vectors.astype(np.float32) @ means.T
        return cls(
            embeddings=means,
            labels=tuple(labels),
            score_min=float(scores.min()),
            score_max=float(scores.max()),
            metadata=metadata or {},
        )
