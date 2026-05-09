"""Adapter protocols + reference implementations for vector DB / embedder / LLM."""

from .embedder import Embedder
from .llm import LLMClient, OpenAICompatLLM
from .vector_db import Candidate, Closeable, VectorBackend

__all__ = [
    "Candidate",
    "Closeable",
    "Embedder",
    "LLMClient",
    "OpenAICompatLLM",
    "VectorBackend",
]
