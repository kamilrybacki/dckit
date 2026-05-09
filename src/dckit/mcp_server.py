"""MCP server exposing dckit operations.

Tools surfaced
--------------
codebook_info     — labels, K, dim, content hash, metadata
classify_text     — tag arbitrary text via embedder + tagger
classify_vector   — tag a raw vector
select_diverse    — apply AreaMMR over caller-supplied candidates
oracle_evaluate   — run LLM-judge on N points sampled from backend

Install
-------
``pip install dckit[mcp]`` brings in the official ``mcp`` Python package.

Run as stdio
------------
::

    python -m dckit.mcp_server --codebook codebook.npz \\
        --qdrant-url http://localhost:6333 --collection vault_merged \\
        --llm-base http://omniroute:30787 --judge-model claude-sonnet-4.5

The server is intentionally LLM- and embedder-agnostic via wiring callbacks
in `serve()`. For a CLI launcher, see `examples/mcp_launcher.py`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .adapters.embedder import Embedder
from .adapters.llm import LLMClient
from .adapters.vector_db import Candidate, VectorBackend
from .codebook import Codebook
from .exceptions import AdapterError
from .oracle import OracleJudge
from .selector import AreaMMR
from .tagger import Tagger

_logger = logging.getLogger(__name__)


EmbedFn = Callable[[list[str]], Any]


@dataclass
class ServerWiring:
    """Bundle of dependencies the MCP server delegates to.

    All fields except ``codebook`` are optional — features that need a missing
    dependency raise a clear ``AdapterError`` at call time rather than failing
    server start.
    """

    codebook: Codebook
    backend: VectorBackend | None = None
    embedder: Embedder | EmbedFn | None = None
    llm: LLMClient | None = None
    judge_model: str | None = None
    text_payload_key: str = "text"


def _embed(emb: Embedder | EmbedFn, texts: list[str]) -> Any:
    if hasattr(emb, "embed"):
        return emb.embed(texts)
    return emb(texts)


def build_server(wiring: ServerWiring) -> Any:
    """Build an MCP `Server` instance bound to the given wiring.

    Returns the raw server object so callers can pick stdio or SSE transport.
    """
    try:
        from mcp.server import Server
        from mcp.types import TextContent, Tool
    except ImportError as exc:
        raise AdapterError(
            "mcp package not installed; pip install dckit[mcp]"
        ) from exc

    tagger = Tagger(wiring.codebook)
    server: Any = Server("dckit")

    @server.list_tools()  # type: ignore[misc]
    async def _list_tools() -> list[Any]:
        return [
            Tool(
                name="codebook_info",
                description="Return labels, K, dim, content hash, metadata of the loaded codebook.",
                inputSchema={"type": "object", "properties": {}, "required": []},
            ),
            Tool(
                name="classify_text",
                description="Embed text(s) and return per-text top-k codebook areas with scores.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "texts": {"type": "array", "items": {"type": "string"}},
                        "top_k": {"type": "integer", "minimum": 1, "maximum": 16, "default": 3},
                    },
                    "required": ["texts"],
                },
            ),
            Tool(
                name="classify_vector",
                description="Tag a raw embedding vector. Returns top-k areas with scores.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "vector": {"type": "array", "items": {"type": "number"}},
                        "top_k": {"type": "integer", "minimum": 1, "maximum": 16, "default": 3},
                    },
                    "required": ["vector"],
                },
            ),
            Tool(
                name="select_diverse",
                description=(
                    "Apply AreaMMR over a list of candidates. "
                    "Each candidate must be {id, score, l1_argmax}."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "candidates": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string"},
                                    "score": {"type": "number"},
                                    "l1_argmax": {"type": "integer"},
                                },
                                "required": ["id", "score", "l1_argmax"],
                            },
                        },
                        "k": {"type": "integer", "minimum": 1, "default": 10},
                        "gamma": {"type": "number", "default": 0.10},
                        "lambda_": {"type": "number", "default": 0.7},
                    },
                    "required": ["candidates"],
                },
            ),
            Tool(
                name="oracle_evaluate",
                description=(
                    "Run LLM-as-judge oracle on N samples from the backend. "
                    "Returns accuracy + per-area breakdown."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "n_samples": {"type": "integer", "minimum": 10, "default": 100},
                        "passage_chars": {"type": "integer", "default": 800},
                    },
                    "required": [],
                },
            ),
        ]

    @server.call_tool()  # type: ignore[misc]
    async def _call_tool(name: str, arguments: dict[str, Any]) -> list[Any]:
        result = await _dispatch(name, arguments, wiring, tagger)
        return [TextContent(type="text", text=_to_json(result))]

    return server


async def _dispatch(
    name: str,
    args: dict[str, Any],
    wiring: ServerWiring,
    tagger: Tagger,
) -> dict[str, Any]:
    if name == "codebook_info":
        return _codebook_info(wiring.codebook)
    if name == "classify_text":
        return await _classify_text(args, wiring, tagger)
    if name == "classify_vector":
        return _classify_vector(args, wiring, tagger)
    if name == "select_diverse":
        return _select_diverse(args)
    if name == "oracle_evaluate":
        return await _oracle_evaluate(args, wiring)
    raise AdapterError(f"unknown tool: {name}")


def _codebook_info(cb: Codebook) -> dict[str, Any]:
    return {
        "k": cb.k,
        "dim": cb.dim,
        "labels": list(cb.labels),
        "content_hash": cb.content_hash(),
        "min_val": cb.min_val,
        "max_val": cb.max_val,
        "metadata": cb.metadata,
    }


async def _classify_text(
    args: dict[str, Any],
    wiring: ServerWiring,
    tagger: Tagger,
) -> dict[str, Any]:
    if wiring.embedder is None:
        raise AdapterError("classify_text needs an embedder; pass embedder= when building server")
    texts: list[str] = list(args.get("texts") or [])
    top_k = int(args.get("top_k", 3))
    import numpy as np

    vectors = np.asarray(_embed(wiring.embedder, texts), dtype=np.float32)
    scores = tagger.project_normalised(vectors)
    out = []
    for text, row in zip(texts, scores, strict=False):
        top_indices = row.argsort()[::-1][:top_k]
        out.append(
            {
                "text": text,
                "top": [
                    {
                        "index": int(i),
                        "label": tagger.labels_for(np.array([int(i)]))[0],
                        "score": round(float(row[i]), 4),
                    }
                    for i in top_indices
                ],
            }
        )
    return {"results": out}


def _classify_vector(
    args: dict[str, Any],
    wiring: ServerWiring,
    tagger: Tagger,
) -> dict[str, Any]:
    import numpy as np

    vec = np.asarray(args["vector"], dtype=np.float32)
    if vec.ndim != 1 or vec.shape[0] != wiring.codebook.dim:
        raise AdapterError(f"vector must be 1-D with {wiring.codebook.dim} entries")
    top_k = int(args.get("top_k", 3))
    scores = tagger.project_normalised(vec[None, :])[0]
    top_indices = scores.argsort()[::-1][:top_k]
    return {
        "top": [
            {
                "index": int(i),
                "label": tagger.labels_for(np.array([int(i)]))[0],
                "score": round(float(scores[i]), 4),
            }
            for i in top_indices
        ],
    }


def _select_diverse(args: dict[str, Any]) -> dict[str, Any]:
    raw: list[dict[str, Any]] = list(args.get("candidates") or [])
    k = int(args.get("k", 10))
    gamma = float(args.get("gamma", 0.10))
    lambda_ = float(args.get("lambda_", 0.7))
    candidates = [
        Candidate(
            id=str(c["id"]),
            score=float(c["score"]),
            payload={"l1_argmax": int(c["l1_argmax"])},
        )
        for c in raw
    ]
    selector = AreaMMR(lambda_=lambda_, gamma=gamma)
    chosen = selector.select(candidates, k)
    return {
        "selected": [
            {"id": c.id, "score": c.score, "l1_argmax": c.payload["l1_argmax"]}
            for c in chosen
        ],
    }


async def _oracle_evaluate(
    args: dict[str, Any],
    wiring: ServerWiring,
) -> dict[str, Any]:
    if wiring.backend is None or wiring.llm is None or not wiring.judge_model:
        raise AdapterError(
            "oracle_evaluate needs backend, llm, and judge_model wired into ServerWiring"
        )
    n_samples = int(args.get("n_samples", 100))
    passage_chars = int(args.get("passage_chars", 800))
    judge = OracleJudge(
        llm=wiring.llm, codebook=wiring.codebook, judge_model=wiring.judge_model
    )
    report = judge.evaluate(
        wiring.backend,
        n_samples=n_samples,
        text_key=wiring.text_payload_key,
        passage_chars=passage_chars,
    )
    return report.to_dict()


def _to_json(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False, indent=2)


async def serve_stdio(wiring: ServerWiring) -> None:
    """Run the MCP server over stdio. Blocks until stdin closes."""
    try:
        from mcp.server.stdio import stdio_server
    except ImportError as exc:
        raise AdapterError(
            "mcp package not installed; pip install dckit[mcp]"
        ) from exc
    server = build_server(wiring)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


__all__ = [
    "ServerWiring",
    "build_server",
    "serve_stdio",
]
