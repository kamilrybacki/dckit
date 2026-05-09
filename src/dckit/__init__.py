"""dckit — discrete-codebook taxonomy middleware for local vector-DB knowledge bases."""

from .codebook import Codebook
from .discovery import DiscoveryConfig, DiscoveryReport, DiscoveryRun, discover
from .exceptions import (
    AdapterError,
    CodebookError,
    DckitError,
    DiscoveryError,
    LLMResponseError,
    OracleError,
)
from .oracle import JudgeVerdict, OracleJudge, OracleReport
from .selector import AreaMMR, FullMMR
from .tagger import Tagger, TagResult

__version__ = "0.1.0"

__all__ = [
    "AdapterError",
    "AreaMMR",
    "Codebook",
    "CodebookError",
    "DckitError",
    "DiscoveryConfig",
    "DiscoveryError",
    "DiscoveryReport",
    "DiscoveryRun",
    "FullMMR",
    "JudgeVerdict",
    "LLMResponseError",
    "OracleError",
    "OracleJudge",
    "OracleReport",
    "TagResult",
    "Tagger",
    "__version__",
    "discover",
]
