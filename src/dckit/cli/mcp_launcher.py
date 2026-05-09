"""CLI: launch dckit MCP server over stdio.

Usage::

    dckit-mcp \\
        --codebook codebook.npz \\
        --qdrant-url http://localhost:6333 \\
        --collection vault_merged \\
        --llm-base http://omniroute:30787 \\
        --llm-key-env OMNIROUTE_API_KEY \\
        --judge-model claude-sonnet-4.5 \\
        --embed-model BAAI/bge-small-en-v1.5

`--embed-model` enables `classify_text`. Without it, the server starts but
text classification raises AdapterError.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from typing import Any

from ..adapters.llm import OpenAICompatLLM
from ..adapters.vector_db import QdrantBackend
from ..codebook import Codebook
from ..mcp_server import ServerWiring, serve_stdio

_logger = logging.getLogger("dckit.mcp")


def _build_embedder(model_name: str | None) -> Any:
    if not model_name:
        return None
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        _logger.warning(
            "sentence-transformers not installed; classify_text will raise. "
            "Install with: pip install sentence-transformers"
        )
        return None
    model = SentenceTransformer(model_name)

    class _ST:
        @property
        def dim(self) -> int:
            return int(model.get_sentence_embedding_dimension())

        def embed(self, texts: list[str]) -> Any:
            return model.encode(texts, normalize_embeddings=True)

    return _ST()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="dckit-mcp", description=__doc__)
    p.add_argument("--codebook", required=True, help="path to codebook .npz")
    p.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    p.add_argument("--qdrant-key-env", default=None, help="env var holding Qdrant API key")
    p.add_argument("--collection", default=os.getenv("QDRANT_COLLECTION", "vault_merged"))
    p.add_argument("--llm-base", default=os.getenv("LLM_BASE_URL"))
    p.add_argument("--llm-key-env", default=None, help="env var holding LLM API key")
    p.add_argument("--judge-model", default=os.getenv("DCKIT_JUDGE_MODEL"))
    p.add_argument("--embed-model", default=os.getenv("DCKIT_EMBED_MODEL"))
    p.add_argument("--text-key", default="text")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )

    codebook = Codebook.load(args.codebook)
    qdrant_key = os.environ[args.qdrant_key_env] if args.qdrant_key_env else None
    backend = QdrantBackend(args.collection, url=args.qdrant_url, api_key=qdrant_key)
    llm: OpenAICompatLLM | None = None
    if args.llm_base:
        llm_key = os.environ[args.llm_key_env] if args.llm_key_env else None
        llm = OpenAICompatLLM(args.llm_base, api_key=llm_key)
    embedder = _build_embedder(args.embed_model)

    wiring = ServerWiring(
        codebook=codebook,
        backend=backend,
        embedder=embedder,
        llm=llm,
        judge_model=args.judge_model,
        text_payload_key=args.text_key,
    )
    _logger.info(
        "starting dckit MCP: codebook=%s K=%d collection=%s",
        codebook.content_hash(),
        codebook.k,
        args.collection,
    )
    asyncio.run(serve_stdio(wiring))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
