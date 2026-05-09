"""Auto-discover a discrete codebook from a vector-DB corpus.

Pipeline (per (model, seed) run):
  1. SAMPLE       stratified pool of points from the backend
  2. MICRO-K      mini-batch k-means → K_micro clusters
  3. NAME         LLM names each micro-cluster (3 medoid excerpts → noun phrase)
  4. PROPOSE      LLM proposes K_macro (code, title) pairs
  5. ASSIGN       embed labels, assign micro→macro by cosine similarity
  6. REFINE       replace empty areas via LLM, re-assign
  7. FIT          build Codebook from class means

Cross-product `models × seeds` produces N runs; the run with the highest
composite score (ARI vs other runs + populated_ratio) is returned.

The propose-then-assign decoupling is the load-bearing trick: LLMs name
*areas* well but assign *members* poorly. Embeddings flip those strengths.
"""

from __future__ import annotations

import json
import logging
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._internal.json_extract import extract_json_array
from ._internal.prompts import (
    MACRO_PROPOSE_PROMPT,
    MICRO_NAME_PROMPT,
    REFINE_EMPTY_PROMPT,
)
from .adapters.embedder import Embedder
from .adapters.llm import LLMClient, parse_json_response, strip_fences
from .adapters.vector_db import Candidate, VectorBackend, iter_all
from .codebook import Codebook
from .exceptions import DiscoveryError

_logger = logging.getLogger(__name__)

DEFAULT_CODES = (
    "00", "05", "10", "15", "20", "25", "30", "35", "40",
    "45", "50", "55", "60", "65", "70", "75", "80", "85", "90",
)


EmbedFn = Callable[[list[str]], np.ndarray]


@dataclass(frozen=True)
class DiscoveryConfig:
    k_macro: int = 16
    k_micro: int = 50
    pool_size: int = 30_000
    sample_size: int = 3_000
    text_payload_key: str = "text"
    stratify_payload_key: str | None = None
    codes: tuple[str, ...] = DEFAULT_CODES
    refine_rounds: int = 2
    micro_excerpt_chars: int = 250


@dataclass(frozen=True)
class DiscoveryRun:
    """Result of one (model, seed) discovery pass."""

    codebook: Codebook
    model: str
    seed: int
    populated_ratio: float
    micro_names: tuple[str, ...]
    assignments: tuple[int, ...]
    composite_score: float = 0.0


@dataclass
class DiscoveryReport:
    best: DiscoveryRun
    runs: list[DiscoveryRun] = field(default_factory=list)


def _embed_callable(embedder: Embedder | EmbedFn) -> EmbedFn:
    if callable(embedder) and not hasattr(embedder, "embed"):
        return embedder  # plain callable
    return embedder.embed  # type: ignore[union-attr,return-value]


def _sample_points(
    backend: VectorBackend,
    cfg: DiscoveryConfig,
    rng: random.Random,
) -> list[Candidate]:
    """Pull a bounded pool, then random-sample with vectors+text."""
    pool: list[Candidate] = []
    payload_keys = [cfg.text_payload_key]
    if cfg.stratify_payload_key:
        payload_keys.append(cfg.stratify_payload_key)
    for cand in iter_all(
        backend, batch=2000, with_payload=payload_keys, with_vectors=False
    ):
        pool.append(cand)
        if len(pool) >= cfg.pool_size:
            break
    if not pool:
        raise DiscoveryError("backend returned zero points; cannot discover")
    n = min(cfg.sample_size, len(pool))
    sampled_meta = rng.sample(pool, n)
    sampled_ids = [c.id for c in sampled_meta]
    # Re-fetch with vectors. Use scroll filter where supported; fallback: chunked queries
    # via repeated single-id retrieval is backend-specific, so we lean on the caller to
    # have a backend.scroll that returns vectors when asked.
    by_id: dict[str, Candidate] = {}
    for cand in iter_all(
        backend,
        batch=2000,
        with_payload=[cfg.text_payload_key],
        with_vectors=True,
    ):
        if cand.id in {c.id for c in sampled_meta}:
            by_id[cand.id] = cand
            if len(by_id) == len(sampled_ids):
                break
    out = [by_id[c.id] for c in sampled_meta if c.id in by_id]
    if not out:
        raise DiscoveryError("could not fetch vectors for sampled points")
    _logger.info("sampled %d points (pool=%d)", len(out), len(pool))
    return out


def _micro_cluster(
    samples: list[Candidate],
    k_micro: int,
    seed: int,
) -> tuple[np.ndarray, list[list[int]]]:
    from sklearn.cluster import MiniBatchKMeans

    vectors = np.stack([s.vector for s in samples if s.vector is not None]).astype(np.float32)
    if len(vectors) < k_micro:
        raise DiscoveryError(
            f"only {len(vectors)} vectors available, need ≥ k_micro={k_micro}"
        )
    km = MiniBatchKMeans(
        n_clusters=k_micro,
        random_state=seed,
        n_init=10,
        batch_size=256,
        max_iter=200,
    )
    labels = km.fit_predict(vectors)
    members: list[list[int]] = [[] for _ in range(k_micro)]
    for idx, lab in enumerate(labels):
        members[int(lab)].append(idx)
    return km.cluster_centers_, members


def _medoid_excerpts(
    samples: list[Candidate],
    centroid: np.ndarray,
    member_indices: list[int],
    text_key: str,
    n: int = 3,
    char_limit: int = 250,
) -> list[str]:
    if not member_indices:
        return []
    vecs = np.stack([samples[i].vector for i in member_indices]).astype(np.float32)
    dists = np.linalg.norm(vecs - centroid[None, :], axis=1)
    order = np.argsort(dists)[:n]
    out: list[str] = []
    for o in order:
        s = samples[member_indices[int(o)]]
        text = (s.payload.get(text_key) or "")[:char_limit] if s.payload else ""
        if text:
            out.append(text)
    return out


def _name_micro(
    llm: LLMClient,
    model: str,
    samples: list[Candidate],
    centroids: np.ndarray,
    members: list[list[int]],
    text_key: str,
    excerpt_chars: int,
) -> list[str]:
    names: list[str] = []
    for i, (c, m) in enumerate(zip(centroids, members, strict=False)):
        excerpts = _medoid_excerpts(samples, c, m, text_key, 3, excerpt_chars)
        if not excerpts:
            names.append(f"empty_cluster_{i}")
            continue
        prompt = MICRO_NAME_PROMPT.format(
            c1=excerpts[0],
            c2=excerpts[1] if len(excerpts) > 1 else "",
            c3=excerpts[2] if len(excerpts) > 2 else "",
        )
        try:
            raw = llm.chat(prompt, model=model, max_tokens=40, temperature=0.3)
        except (OSError, RuntimeError) as exc:
            _logger.warning("micro-name %d/%d failed: %s", i + 1, len(centroids), exc)
            names.append(f"unnamed_{i}")
            continue
        cleaned = strip_fences(raw).splitlines()[0].strip(' "\'.')
        names.append(cleaned[:80] or f"unnamed_{i}")
    return names


def _propose_macros(
    llm: LLMClient,
    model: str,
    micro_names: list[str],
    cfg: DiscoveryConfig,
) -> list[dict[str, str]]:
    listing = "\n".join(f"  - {nm}" for nm in micro_names)
    prompt = MACRO_PROPOSE_PROMPT.format(
        k_macro=cfg.k_macro,
        codes_csv=", ".join(cfg.codes),
        n_micro=len(micro_names),
        micro_topics_listing=listing,
    )
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            raw = llm.chat(prompt, model=model, max_tokens=1200, temperature=0.2)
            arr = extract_json_array(raw)
            _validate_proposal(arr, cfg.k_macro, cfg.codes)
            return [{"code": str(it["code"]), "title": str(it["title"])} for it in arr]
        except (json.JSONDecodeError, ValueError) as exc:
            last_exc = exc
            _logger.warning("macro propose attempt %d/3 invalid: %s", attempt + 1, exc)
    raise DiscoveryError(f"macro propose failed after 3 attempts: {last_exc}")


def _validate_proposal(
    arr: list[dict[str, Any]],
    k_macro: int,
    codes: tuple[str, ...],
) -> None:
    if len(arr) != k_macro:
        raise ValueError(f"expected {k_macro} areas, got {len(arr)}")
    seen: set[str] = set()
    last = -1
    for item in arr:
        code = str(item.get("code", ""))
        title = str(item.get("title", "")).strip()
        if code not in codes:
            raise ValueError(f"invalid code {code!r}")
        if code in seen:
            raise ValueError(f"duplicate code {code!r}")
        if int(code) <= last:
            raise ValueError(f"codes not ascending at {code!r}")
        if not title:
            raise ValueError(f"missing title for {code!r}")
        last = int(code)
        seen.add(code)


def _assign_micros(
    macros: list[dict[str, str]],
    micro_names: list[str],
    embed: EmbedFn,
) -> list[dict[str, Any]]:
    macro_texts = [f"{m['code']} — {m['title']}" for m in macros]
    macro_vecs = np.asarray(embed(macro_texts), dtype=np.float32)
    micro_vecs = np.asarray(embed(micro_names), dtype=np.float32)
    macro_unit = macro_vecs / (np.linalg.norm(macro_vecs, axis=1, keepdims=True) + 1e-10)
    micro_unit = micro_vecs / (np.linalg.norm(micro_vecs, axis=1, keepdims=True) + 1e-10)
    sims = micro_unit @ macro_unit.T
    assignments = sims.argmax(axis=1)
    out = [{"code": m["code"], "title": m["title"], "members": []} for m in macros]
    for mi, ai in enumerate(assignments):
        out[int(ai)]["members"].append(int(mi))
    return out


def _refine_empty(
    llm: LLMClient,
    model: str,
    areas: list[dict[str, Any]],
    micro_names: list[str],
    embed: EmbedFn,
    cfg: DiscoveryConfig,
) -> list[dict[str, Any]]:
    for round_idx in range(cfg.refine_rounds):
        empty = [a for a in areas if not a["members"]]
        if not empty:
            return areas
        populated = [a for a in areas if a["members"]]
        used = {a["code"] for a in populated}
        avail = [c for c in cfg.codes if c not in used]
        if len(avail) < len(empty):
            _logger.warning("refine round %d: not enough free codes", round_idx + 1)
            return areas
        empty_listing = "\n".join(f"  {a['code']} — {a['title']}" for a in empty)
        populated_listing = "\n".join(
            f"  {a['code']} — {a['title']}: "
            + ", ".join(micro_names[i] for i in a["members"][:3])
            for a in populated
        )
        prompt = REFINE_EMPTY_PROMPT.format(
            n_empty=len(empty),
            empty_listing=empty_listing,
            populated_listing=populated_listing,
            available_codes=", ".join(avail),
        )
        try:
            raw = llm.chat(prompt, model=model, max_tokens=600, temperature=0.3)
            replacements = extract_json_array(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            _logger.warning("refine round %d failed: %s", round_idx + 1, exc)
            return areas
        if len(replacements) != len(empty):
            _logger.warning(
                "refine round %d: expected %d replacements, got %d — keeping current areas",
                round_idx + 1, len(empty), len(replacements),
            )
            return areas
        merged = [{"code": a["code"], "title": a["title"]} for a in populated]
        merged.extend({"code": str(r["code"]), "title": str(r["title"])} for r in replacements)
        merged.sort(key=lambda x: int(x["code"]))
        areas = _assign_micros(merged, micro_names, embed)
    return areas


def _fit_codebook(
    samples: list[Candidate],
    micro_assignments: list[int],
    areas: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> Codebook:
    """Build codebook embeddings from points whose micro cluster maps to each area.

    Empty macros (no member micros) are dropped — the resulting K is the count
    of populated areas. The original full-K layout is preserved in metadata.
    """
    populated = [a for a in areas if a["members"]]
    if not populated:
        raise DiscoveryError("no populated areas after assignment")
    macro_to_idx = {id(a): i for i, a in enumerate(populated)}
    micro_to_macro: dict[int, int] = {}
    for a in populated:
        for mi in a["members"]:
            micro_to_macro[mi] = macro_to_idx[id(a)]
    point_assignments = np.array(
        [micro_to_macro.get(int(m), -1) for m in micro_assignments],
        dtype=np.int64,
    )
    vectors = np.stack([s.vector for s in samples if s.vector is not None]).astype(np.float32)
    if len(vectors) != len(point_assignments):
        raise DiscoveryError("vector / assignment length mismatch")
    keep_mask = point_assignments >= 0
    vectors = vectors[keep_mask]
    point_assignments = point_assignments[keep_mask]
    labels = [f"{a['code']} — {a['title']}" for a in populated]
    full_metadata = {
        **metadata,
        "k_full": len(areas),
        "k_populated": len(populated),
        "dropped_codes": [a["code"] for a in areas if not a["members"]],
    }
    return Codebook.from_class_means(vectors, point_assignments, labels, metadata=full_metadata)


def _composite_score(populated_ratio: float, ari_mean: float) -> float:
    return 0.5 * ari_mean + 0.5 * populated_ratio


def discover(
    backend: VectorBackend,
    embedder: Embedder | EmbedFn,
    llm: LLMClient,
    *,
    models: list[str],
    seeds: list[int] | None = None,
    config: DiscoveryConfig | None = None,
) -> DiscoveryReport:
    """Run discovery across (models × seeds), select best by composite score.

    Parameters
    ----------
    backend, embedder, llm:
        Adapter instances satisfying the relevant Protocol.
    models:
        LLM model names to try (e.g. ["claude-haiku-4.5", "deepseek-chat"]).
    seeds:
        K-means RNG seeds. Defaults to [0, 1].
    config:
        Discovery hyperparameters. Defaults to `DiscoveryConfig()`.
    """
    cfg = config or DiscoveryConfig()
    seeds = seeds or [0, 1]
    embed = _embed_callable(embedder)
    rng = random.Random(0)

    samples = _sample_points(backend, cfg, rng)
    runs: list[DiscoveryRun] = []
    micro_assignments_per_seed: dict[int, list[int]] = {}

    for seed in seeds:
        centroids, members = _micro_cluster(samples, cfg.k_micro, seed)
        micro_assignments = [-1] * len(samples)
        for ci, mlist in enumerate(members):
            for idx in mlist:
                micro_assignments[idx] = ci
        micro_assignments_per_seed[seed] = micro_assignments

        for model in models:
            try:
                names = _name_micro(
                    llm, model, samples, centroids, members,
                    cfg.text_payload_key, cfg.micro_excerpt_chars,
                )
                proposal = _propose_macros(llm, model, names, cfg)
                areas = _assign_micros(proposal, names, embed)
                areas = _refine_empty(llm, model, areas, names, embed, cfg)
                populated = sum(1 for a in areas if a["members"])
                populated_ratio = populated / cfg.k_macro
                codebook = _fit_codebook(
                    samples,
                    micro_assignments,
                    areas,
                    metadata={
                        "model": model,
                        "seed": seed,
                        "k_macro": cfg.k_macro,
                        "k_micro": cfg.k_micro,
                        "n_samples": len(samples),
                    },
                )
                runs.append(
                    DiscoveryRun(
                        codebook=codebook,
                        model=model,
                        seed=seed,
                        populated_ratio=populated_ratio,
                        micro_names=tuple(names),
                        assignments=tuple(micro_assignments),
                    )
                )
                _logger.info(
                    "run model=%s seed=%d populated=%d/%d", model, seed,
                    populated, cfg.k_macro,
                )
            except DiscoveryError as exc:
                _logger.warning("run model=%s seed=%d failed: %s", model, seed, exc)

    if not runs:
        raise DiscoveryError("all (model, seed) runs failed")

    scored = _score_runs(runs, samples)
    best = max(scored, key=lambda r: r.composite_score)
    return DiscoveryReport(best=best, runs=scored)


def _score_runs(runs: list[DiscoveryRun], samples: list[Candidate]) -> list[DiscoveryRun]:
    """Compute mean ARI of each run vs all others; combine with populated_ratio."""
    from sklearn.metrics import adjusted_rand_score

    n = len(runs)
    if n == 1:
        return [
            DiscoveryRun(
                codebook=runs[0].codebook,
                model=runs[0].model,
                seed=runs[0].seed,
                populated_ratio=runs[0].populated_ratio,
                micro_names=runs[0].micro_names,
                assignments=runs[0].assignments,
                composite_score=runs[0].populated_ratio,
            )
        ]
    # Use codebook tag of each sample as the assignment for ARI comparison.
    point_assignments_per_run: list[np.ndarray] = []
    vectors = np.stack(
        [s.vector for s in samples if s.vector is not None]
    ).astype(np.float32)
    for r in runs:
        scores = vectors @ r.codebook.embeddings.T
        point_assignments_per_run.append(scores.argmax(axis=1))
    out: list[DiscoveryRun] = []
    for i, r in enumerate(runs):
        ari_vals = [
            adjusted_rand_score(point_assignments_per_run[i], point_assignments_per_run[j])
            for j in range(n)
            if j != i
        ]
        ari_mean = float(np.mean(ari_vals)) if ari_vals else 0.0
        score = _composite_score(r.populated_ratio, ari_mean)
        out.append(
            DiscoveryRun(
                codebook=r.codebook,
                model=r.model,
                seed=r.seed,
                populated_ratio=r.populated_ratio,
                micro_names=r.micro_names,
                assignments=r.assignments,
                composite_score=score,
            )
        )
    return out


__all__ = [
    "DiscoveryConfig",
    "DiscoveryReport",
    "DiscoveryRun",
    "discover",
]


# Suppress unused warnings for parse_json_response (re-exported for tests)
_ = parse_json_response
