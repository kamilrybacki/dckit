"""LLM-as-judge oracle for codebook quality.

Samples N points from the corpus, asks the LLM to pick the best area for each
passage, compares against the codebook's argmax tag. Reports per-area
accuracy + a confident-bucket subset where LLM marked confidence=high.

Empirical baseline: random 1-of-16 = 0.0625. A sound codebook should
score > 0.40; production-grade ≥ 0.50.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._internal.json_extract import extract_json_object
from ._internal.prompts import ORACLE_JUDGE_PROMPT
from .adapters.llm import LLMClient, strip_fences
from .adapters.vector_db import Candidate, VectorBackend, iter_all
from .codebook import Codebook
from .exceptions import OracleError
from .tagger import Tagger

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class JudgeVerdict:
    """Single LLM judgement on one passage."""

    point_id: str
    codebook_pick: int
    judge_pick: int
    confidence: str
    agree: bool


@dataclass
class OracleReport:
    """Aggregate result of an oracle evaluation."""

    n_evaluated: int
    n_agreement: int
    accuracy: float
    confident_n: int
    confident_accuracy: float
    per_area_accuracy: dict[int, float]
    verdicts: list[JudgeVerdict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_evaluated": self.n_evaluated,
            "accuracy": self.accuracy,
            "confident_n": self.confident_n,
            "confident_accuracy": self.confident_accuracy,
            "per_area_accuracy": self.per_area_accuracy,
        }


class OracleJudge:
    """LLM-as-judge harness for codebook validation.

    Usage::

        judge = OracleJudge(llm=my_llm, codebook=cb, judge_model="claude-sonnet-4.5")
        report = judge.evaluate(backend, n_samples=500, text_key="text")
        assert report.accuracy > 0.45
    """

    def __init__(
        self,
        *,
        llm: LLMClient,
        codebook: Codebook,
        judge_model: str,
        max_tokens: int = 80,
    ) -> None:
        self._llm = llm
        self._codebook = codebook
        self._tagger = Tagger(codebook)
        self._model = judge_model
        self._max_tokens = max_tokens

    def evaluate(
        self,
        backend: VectorBackend,
        *,
        n_samples: int,
        text_key: str = "text",
        passage_chars: int = 800,
        seed: int = 0,
    ) -> OracleReport:
        """Sample N points, judge each, return aggregate report."""
        import random

        rng = random.Random(seed)
        pool: list[Candidate] = []
        for cand in iter_all(backend, batch=2000, with_payload=[text_key], with_vectors=True):
            pool.append(cand)
        if len(pool) < n_samples:
            raise OracleError(f"backend has {len(pool)} points, need ≥ {n_samples}")
        sampled = rng.sample(pool, n_samples)
        verdicts: list[JudgeVerdict] = []
        per_area_hits: Counter[int] = Counter()
        per_area_total: Counter[int] = Counter()
        for s in sampled:
            if s.vector is None:
                continue
            text = (s.payload.get(text_key) or "")[:passage_chars] if s.payload else ""
            if not text.strip():
                continue
            cb_pick = int(self._tagger.tag(s.vector[None, :])[0])
            verdict = self._judge_one(s.id, text, cb_pick)
            if verdict is None:
                continue
            verdicts.append(verdict)
            per_area_total[cb_pick] += 1
            if verdict.agree:
                per_area_hits[cb_pick] += 1
        n_eval = len(verdicts)
        if n_eval == 0:
            raise OracleError("zero successful judgements")
        n_agree = sum(1 for v in verdicts if v.agree)
        confident = [v for v in verdicts if v.confidence == "high"]
        per_area_accuracy = {
            area: per_area_hits[area] / per_area_total[area]
            for area in per_area_total
        }
        return OracleReport(
            n_evaluated=n_eval,
            n_agreement=n_agree,
            accuracy=n_agree / n_eval,
            confident_n=len(confident),
            confident_accuracy=(
                sum(1 for v in confident if v.agree) / len(confident)
                if confident
                else 0.0
            ),
            per_area_accuracy=per_area_accuracy,
            verdicts=verdicts,
        )

    def _judge_one(
        self,
        point_id: str,
        passage: str,
        codebook_pick: int,
    ) -> JudgeVerdict | None:
        area_listing = "\n".join(
            f"  [{i}] {label}" for i, label in enumerate(self._codebook.labels)
        )
        prompt = ORACLE_JUDGE_PROMPT.format(
            k=self._codebook.k,
            passage=passage,
            area_listing=area_listing,
        )
        try:
            raw = self._llm.chat(
                prompt,
                model=self._model,
                temperature=0.0,
                max_tokens=self._max_tokens,
            )
            parsed = extract_json_object(strip_fences(raw))
        except (json.JSONDecodeError, ValueError, RuntimeError) as exc:
            _logger.warning("judge skipped %s: %s", point_id, exc)
            return None
        try:
            judge_pick = int(parsed["area_index"])
        except (KeyError, ValueError, TypeError):
            return None
        if not 0 <= judge_pick < self._codebook.k:
            return None
        confidence = str(parsed.get("confidence", "medium")).lower()
        if confidence not in {"high", "medium", "low"}:
            confidence = "medium"
        return JudgeVerdict(
            point_id=point_id,
            codebook_pick=codebook_pick,
            judge_pick=judge_pick,
            confidence=confidence,
            agree=(judge_pick == codebook_pick),
        )


# Avoid unused import warning — np is used indirectly via Tagger
_ = np
