"""LLM-judge oracle with stub LLM."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from dckit.codebook import Codebook
from dckit.oracle import OracleJudge


class AlwaysPickZero:
    """LLM that always picks area 0 with high confidence."""

    def chat(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        system: str | None = None,
    ) -> str:
        return '{"area_index": 0, "confidence": "high"}'


@pytest.mark.unit
def test_oracle_evaluate_runs_end_to_end(synthetic_corpus: Any, fake_backend: Any) -> None:
    rng = np.random.default_rng(0)
    means = rng.standard_normal((4, 16)).astype(np.float32)
    means /= np.linalg.norm(means, axis=1, keepdims=True)
    cb = Codebook(means, ("00", "10", "20", "30"), -1.0, 1.0, {})
    judge = OracleJudge(llm=AlwaysPickZero(), codebook=cb, judge_model="stub")
    backend = fake_backend(synthetic_corpus)
    report = judge.evaluate(backend, n_samples=20, text_key="text")
    assert report.n_evaluated > 0
    assert 0.0 <= report.accuracy <= 1.0
    assert report.confident_n == report.n_evaluated  # all high confidence
    assert all(v.confidence == "high" for v in report.verdicts)


@pytest.mark.unit
def test_oracle_handles_garbage_llm_output(synthetic_corpus: Any, fake_backend: Any) -> None:
    class GarbageLLM:
        def chat(self, prompt: str, **_: Any) -> str:
            return "I refuse."

    rng = np.random.default_rng(0)
    means = rng.standard_normal((4, 16)).astype(np.float32)
    means /= np.linalg.norm(means, axis=1, keepdims=True)
    cb = Codebook(means, ("00", "10", "20", "30"), -1.0, 1.0, {})
    judge = OracleJudge(llm=GarbageLLM(), codebook=cb, judge_model="stub")
    backend = fake_backend(synthetic_corpus)
    from dckit.exceptions import OracleError

    with pytest.raises(OracleError):
        judge.evaluate(backend, n_samples=10, text_key="text")
