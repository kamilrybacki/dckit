"""Exception hierarchy for dckit."""

from __future__ import annotations


class DckitError(Exception):
    """Base for all dckit-raised errors."""


class CodebookError(DckitError):
    """Codebook construction, validation, or io failure."""


class DiscoveryError(DckitError):
    """Auto-discovery pipeline failure."""


class AdapterError(DckitError):
    """Vector-DB / embedder / LLM adapter failure."""


class LLMResponseError(AdapterError):
    """LLM returned malformed or unparsable output."""


class OracleError(DckitError):
    """LLM-judge oracle failure."""
