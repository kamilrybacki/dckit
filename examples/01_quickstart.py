"""Quickstart: build a tiny codebook, tag vectors, do diversity-aware selection.

Runs without any external services — uses synthetic data, FakeBackend, and a
stub LLM. Real workflows substitute QdrantBackend + a real embedder + a real
OpenAI-compatible LLM endpoint (e.g. OmniRoute, OpenRouter, Ollama).
"""

from __future__ import annotations

import logging

import numpy as np

from dckit import AreaMMR, Codebook, Tagger
from dckit.adapters.vector_db import Candidate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    rng = np.random.default_rng(0)
    centers = rng.standard_normal((4, 16)).astype(np.float32) * 3.0
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    cb = Codebook(
        embeddings=centers,
        labels=("00 — Tools", "10 — Concepts", "20 — Workflows", "30 — Reference"),
        min_val=-1.0,
        max_val=1.0,
        metadata={"source": "quickstart"},
    )

    tagger = Tagger(cb)
    candidate_vecs = rng.standard_normal((20, 16)).astype(np.float32)
    candidate_vecs /= np.linalg.norm(candidate_vecs, axis=1, keepdims=True)
    tags = tagger.tag(candidate_vecs)
    candidates = [
        Candidate(
            id=f"chunk-{i}",
            score=float(0.95 - i * 0.01),
            payload={"l1_argmax": int(t)},
        )
        for i, t in enumerate(tags)
    ]

    selector = AreaMMR(gamma=0.20)
    chosen = selector.select(candidates, k=5)

    print(f"codebook hash: {cb.content_hash()}")
    print(f"selected {len(chosen)} from {len(candidates)} candidates")
    for c in chosen:
        print(f"  {c.id}  score={c.score:.3f}  area={cb.label(c.payload['l1_argmax'])}")


if __name__ == "__main__":
    main()
