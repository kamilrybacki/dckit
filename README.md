# dckit — Discrete-Codebook Taxonomy Kit

> Lightweight middleware for diversity-aware retrieval over local vector-DB knowledge bases.

`dckit` extracts the production-validated mechanisms from the
[knovecs](https://github.com/kamilrybacki/knovecs) research project into a
reusable, vector-DB-agnostic Python library:

1. **`Tagger`** — class-mean projection assigns every chunk to a discrete area at ingest time (one matmul, ~1 ms / 1k vectors).
2. **`AreaMMR`** — diversity-aware context selector with a binary categorical penalty. Pareto-optimal vs. full MMR on a 56-vault Obsidian corpus.
3. **`discover`** — propose-then-assign auto-codebook discovery: an LLM proposes K (code, title) pairs; embeddings deterministically assign cluster members.
4. **`OracleJudge`** — LLM-as-judge harness for codebook quality validation.
5. **MCP server** — expose every operation over Model Context Protocol (stdio).

## Install

```bash
pip install dckit                       # core
pip install dckit[qdrant]               # + Qdrant adapter
pip install dckit[mcp]                  # + MCP server
pip install dckit[qdrant,mcp,dev]       # everything
```

## Quickstart

```python
from dckit import AreaMMR, Codebook, Tagger
from dckit.adapters.vector_db import QdrantBackend

# Load a previously-discovered codebook
cb = Codebook.load("codebook.npz")
tagger = Tagger(cb)

# Tag at ingest
tags = tagger.tag(chunk_vectors)            # → ndarray[int64] of area indices
labels = [cb.label(int(t)) for t in tags]   # → ["70 — Systems & Infra", ...]

# Diversity-aware retrieval at query time
backend = QdrantBackend(collection="vault_merged", url="http://localhost:6333")
candidates = backend.query(query_vector, limit=50)
context = AreaMMR(gamma=0.10).select(candidates, k=10)
```

## Auto-discover a codebook

```python
from dckit import discover
from dckit.adapters.llm import OpenAICompatLLM

llm = OpenAICompatLLM("http://omniroute:30787", api_key="...")
report = discover(
    backend=backend,
    embedder=my_embedder,                       # any callable: list[str] → ndarray
    llm=llm,
    models=["claude-haiku-4.5", "deepseek-chat"],
    seeds=[0, 1],
)
report.best.codebook.save("codebook.npz")
```

## Validate codebook quality

```python
from dckit import OracleJudge

judge = OracleJudge(llm=llm, codebook=cb, judge_model="claude-sonnet-4.5")
report = judge.evaluate(backend, n_samples=500)
print(f"agreement: {report.accuracy:.3f}")
assert report.accuracy > 0.45            # baseline: 1/K = 0.0625 for K=16
```

## MCP server

```bash
dckit-mcp \
  --codebook codebook.npz \
  --qdrant-url http://localhost:6333 \
  --collection vault_merged \
  --llm-base http://omniroute:30787 \
  --llm-key-env OMNIROUTE_API_KEY \
  --judge-model claude-sonnet-4.5 \
  --embed-model BAAI/bge-small-en-v1.5
```

Exposes five tools: `codebook_info`, `classify_text`, `classify_vector`, `select_diverse`, `oracle_evaluate`.

## Adapters

`VectorBackend`, `Embedder`, `LLMClient` are runtime-checkable Protocols.
Reference implementation: `QdrantBackend`. Bring your own for Milvus, Weaviate,
pgvector, Chroma — typically ~50 lines per backend.

## Empirical findings driving the design

- **Codebook (not classifier mechanism) is the dominant lever.** Manual codebook → 17% LLM-judge alignment; auto-discovered codebook → 51%.
- **Propose-then-assign decoupling** beats pure-LLM clustering: LLMs name areas well but assign members poorly. Embeddings flip those strengths.
- **AreaMMR γ=0.10** is Pareto-optimal: ~0 cost over base top-K, ~80% of full-MMR diversity gain.
- **Cross-encoder reranking, full MMR, and equilibrium re-embedding were tested and rejected** — net-negative or null effect on retrieval recall.

See the [knovecs paper](https://github.com/kamilrybacki/knovecs/blob/main/docs/papers/) for the full benchmark methodology.

## License

MIT — see `LICENSE`.
