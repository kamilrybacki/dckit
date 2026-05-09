"""Real-world pipeline: Qdrant + sentence-transformers + OmniRoute LLM.

Requires::

    pip install dckit[qdrant] sentence-transformers
    export OMNIROUTE_BASE_URL=http://localhost:30787
    export OMNIROUTE_API_KEY=...
    export QDRANT_URL=http://localhost:6333

Run ingest path::

    python examples/02_qdrant_pipeline.py discover --collection vault_merged
    python examples/02_qdrant_pipeline.py tag --collection vault_merged
    python examples/02_qdrant_pipeline.py validate --collection vault_merged
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np

from dckit import AreaMMR, Codebook, OracleJudge, Tagger, discover
from dckit.adapters.llm import OpenAICompatLLM
from dckit.adapters.vector_db import QdrantBackend, iter_all
from dckit.discovery import DiscoveryConfig

CODEBOOK_PATH = "codebook.npz"


def _embedder(model_name: str = "BAAI/bge-small-en-v1.5"):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    class STEmbedder:
        @property
        def dim(self) -> int:
            return int(model.get_sentence_embedding_dimension())

        def embed(self, texts: list[str]) -> np.ndarray:
            return model.encode(texts, normalize_embeddings=True)

    return STEmbedder()


def cmd_discover(args: argparse.Namespace) -> int:
    backend = QdrantBackend(args.collection, url=os.environ["QDRANT_URL"])
    llm = OpenAICompatLLM(
        os.environ["OMNIROUTE_BASE_URL"],
        api_key=os.environ.get("OMNIROUTE_API_KEY"),
    )
    cfg = DiscoveryConfig(k_macro=args.k_macro, k_micro=args.k_micro)
    report = discover(
        backend=backend,
        embedder=_embedder(),
        llm=llm,
        models=args.models.split(","),
        seeds=[int(s) for s in args.seeds.split(",")],
        config=cfg,
    )
    report.best.codebook.save(args.out)
    print(f"saved {args.out}")
    print(f"  hash: {report.best.codebook.content_hash()}")
    print(f"  populated_ratio: {report.best.populated_ratio:.3f}")
    print(f"  composite: {report.best.composite_score:.3f}")
    return 0


def cmd_tag(args: argparse.Namespace) -> int:
    cb = Codebook.load(args.codebook)
    backend = QdrantBackend(args.collection, url=os.environ["QDRANT_URL"])
    tagger = Tagger(cb)
    batch_ids: list[str] = []
    batch_vecs: list[np.ndarray] = []
    n_done = 0
    for cand in iter_all(backend, batch=2000, with_vectors=True, with_payload=False):
        if cand.vector is None:
            continue
        batch_ids.append(cand.id)
        batch_vecs.append(cand.vector)
        if len(batch_ids) >= 1000:
            n_done += _flush(tagger, backend, batch_ids, batch_vecs, cb)
            batch_ids, batch_vecs = [], []
    if batch_ids:
        n_done += _flush(tagger, backend, batch_ids, batch_vecs, cb)
    print(f"tagged {n_done} points in {args.collection}")
    return 0


def _flush(
    tagger: Tagger,
    backend: QdrantBackend,
    ids: list[str],
    vecs: list[np.ndarray],
    cb: Codebook,
) -> int:
    arr = np.stack(vecs).astype(np.float32)
    tags = tagger.tag(arr)
    for i, idx in zip(ids, tags, strict=False):
        backend.set_payload([i], {"l1_argmax": int(idx), "jd_area": cb.label(int(idx))})
    return len(ids)


def cmd_validate(args: argparse.Namespace) -> int:
    cb = Codebook.load(args.codebook)
    backend = QdrantBackend(args.collection, url=os.environ["QDRANT_URL"])
    llm = OpenAICompatLLM(
        os.environ["OMNIROUTE_BASE_URL"],
        api_key=os.environ.get("OMNIROUTE_API_KEY"),
    )
    judge = OracleJudge(llm=llm, codebook=cb, judge_model=args.judge_model)
    report = judge.evaluate(backend, n_samples=args.n_samples, text_key=args.text_key)
    print(f"accuracy:           {report.accuracy:.3f}")
    print(f"confident_accuracy: {report.confident_accuracy:.3f} (n={report.confident_n})")
    print(f"per-area:           {dict(report.per_area_accuracy)}")
    return 0


def cmd_select(args: argparse.Namespace) -> int:
    backend = QdrantBackend(args.collection, url=os.environ["QDRANT_URL"])
    embedder = _embedder()
    qvec = embedder.embed([args.query])[0]
    candidates = backend.query(
        np.asarray(qvec, dtype=np.float32),
        limit=args.top_c,
        with_payload=True,
    )
    chosen = AreaMMR(gamma=args.gamma).select(candidates, k=args.k)
    for c in chosen:
        print(f"  {c.id} score={c.score:.3f} area={c.payload.get('l1_argmax')}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("discover")
    d.add_argument("--collection", required=True)
    d.add_argument("--models", default="claude-haiku-4.5,deepseek-chat")
    d.add_argument("--seeds", default="0,1")
    d.add_argument("--k-macro", type=int, default=16)
    d.add_argument("--k-micro", type=int, default=50)
    d.add_argument("--out", default=CODEBOOK_PATH)
    d.set_defaults(func=cmd_discover)

    t = sub.add_parser("tag")
    t.add_argument("--collection", required=True)
    t.add_argument("--codebook", default=CODEBOOK_PATH)
    t.set_defaults(func=cmd_tag)

    v = sub.add_parser("validate")
    v.add_argument("--collection", required=True)
    v.add_argument("--codebook", default=CODEBOOK_PATH)
    v.add_argument("--judge-model", default="claude-sonnet-4.5")
    v.add_argument("--n-samples", type=int, default=200)
    v.add_argument("--text-key", default="text")
    v.set_defaults(func=cmd_validate)

    s = sub.add_parser("select")
    s.add_argument("--collection", required=True)
    s.add_argument("--query", required=True)
    s.add_argument("--top-c", type=int, default=50)
    s.add_argument("--k", type=int, default=10)
    s.add_argument("--gamma", type=float, default=0.10)
    s.set_defaults(func=cmd_select)
    return p


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
