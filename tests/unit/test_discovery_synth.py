"""Discovery pipeline on a small synthetic corpus with a stub LLM."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from dckit.discovery import DiscoveryConfig, discover


class FakeEmbedder:
    """Echo-style embedder: text-determined vector for deterministic tests."""

    def __init__(self) -> None:
        self.dim = 16

    def embed(self, texts: list[str]) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(tuple(texts))) % (2**32))
        return rng.standard_normal((len(texts), self.dim)).astype(np.float32)


class ScriptedLLM:
    """Returns canned outputs based on which prompt template fragment matches."""

    def __init__(self, k_macro: int) -> None:
        self.k_macro = k_macro

    def chat(self, prompt: str, *, model: str, **kwargs: Any) -> str:
        if "name topics" in prompt or "Topic:" in prompt:
            return "Synthetic Domain Topic"
        if "subject areas" in prompt:
            arr = [
                {"code": code, "title": f"Area {i}"}
                for i, code in enumerate(
                    ["00", "05", "10", "15", "20", "25", "30", "35"][: self.k_macro]
                )
            ]
            import json

            return json.dumps(arr)
        if "REPLACED" in prompt or "must be REPLACED" in prompt:
            import json

            return json.dumps([{"code": "40", "title": "Refined"}])
        return ""


@pytest.mark.unit
def test_discovery_runs_on_synthetic(synthetic_corpus: Any, fake_backend: Any) -> None:
    backend = fake_backend(synthetic_corpus)
    cfg = DiscoveryConfig(
        k_macro=4,
        k_micro=4,
        pool_size=200,
        sample_size=100,
    )
    report = discover(
        backend=backend,
        embedder=FakeEmbedder(),
        llm=ScriptedLLM(k_macro=4),
        models=["stub-a"],
        seeds=[0],
        config=cfg,
    )
    assert report.best is not None
    # K may shrink if some macros end up empty after assignment.
    assert 1 <= report.best.codebook.k <= cfg.k_macro
    assert report.best.codebook.dim == 16
    assert 0.0 <= report.best.populated_ratio <= 1.0
    assert report.best.codebook.metadata["k_full"] == cfg.k_macro
