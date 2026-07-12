"""medqa_multiagent: MedQA-USMLE multi-agent evaluation harness.

This package currently exposes the pure-logic foundation shared by every
later pipeline variant (V0-V4): run configuration, deterministic dev/test
sampling, the shared strict-regex output parser, and cache-key derivation.
No LLM or network calls are involved in any of it.
"""

from .cache_key import compute_cache_key
from .config import RunConfig
from .official_eval import sample_official_test_set
from .parsing import DEFAULT_VALID_OPTIONS, ParsedAnswer, parse_final_answer
from .sampling import deterministic_sample, sample_dev_set

__all__ = [
    "RunConfig",
    "ParsedAnswer",
    "DEFAULT_VALID_OPTIONS",
    "parse_final_answer",
    "deterministic_sample",
    "sample_dev_set",
    "sample_official_test_set",
    "compute_cache_key",
]
