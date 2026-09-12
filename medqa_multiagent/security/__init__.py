"""Prompt-injection attack/defense evaluation helpers.

This package is intentionally additive: normal V0-V4 callers do not import or
configure any of these classes, while the final-project evaluation harness can
inject attacked views of RAG passages and long-term memory records.
"""

from .attacks import build_attack_payload, deterministic_target_option
from .scenario import (
    AttackFamily,
    AttackPosition,
    AttackSurface,
    DefenseMode,
    ModelTrack,
    SecurityScenario,
)
from .structured import StructuredQuery, compose_struq_prompt, recursive_filter
from .wrappers import InjectedMemoryStore, InjectedRetriever

__all__ = [
    "AttackFamily",
    "AttackPosition",
    "AttackSurface",
    "DefenseMode",
    "InjectedMemoryStore",
    "InjectedRetriever",
    "ModelTrack",
    "SecurityScenario",
    "StructuredQuery",
    "build_attack_payload",
    "compose_struq_prompt",
    "deterministic_target_option",
    "recursive_filter",
]
