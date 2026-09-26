"""Typed, serializable configuration for one security-evaluation condition."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, Optional


class _ValueEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class AttackFamily(_ValueEnum):
    CLEAN = "clean"
    NAIVE = "naive"
    ESCAPE = "escape"
    CONTEXT_IGNORING = "context_ignoring"
    FAKE_COMPLETION = "fake_completion"
    COMBINED = "combined"
    DOCUMENT_AUTHORITY = "document_authority"
    CHAINED_COMPLETION = "chained_completion"
    DENSE_AUTHORITY = "dense_authority"
    FEW_SHOT_DEMONSTRATION = "few_shot_demonstration"
    DELIMITER_NEAR_MISS = "delimiter_near_miss"
    MULTILINGUAL = "multilingual"
    ENCODED = "encoded"


class AttackSurface(_ValueEnum):
    RAG = "rag"
    MEMORY = "memory"


class AttackPosition(_ValueEnum):
    TOP1 = "top1"
    ALL_TOP_K = "all_top_k"


class DefenseMode(_ValueEnum):
    NONE = "none"
    FRONTEND_ONLY = "frontend_only"
    HEURISTIC_GUARD = "heuristic_guard"
    STRUQ = "struq"


class ModelTrack(_ValueEnum):
    API = "api"
    API_HEURISTIC_GUARD = "api_heuristic_guard"
    LOCAL_UNDEFENDED = "local_undefended"
    LOCAL_STRUQ = "local_struq"
    FRONTEND_ONLY = "frontend_only"
    OLLAMA_UNDEFENDED = "ollama_undefended"
    OLLAMA_FRONTEND_ONLY = "ollama_frontend_only"
    OLLAMA_HEURISTIC_GUARD = "ollama_heuristic_guard"


@dataclass(frozen=True)
class SecurityScenario:
    """One pre-registered attack/defense condition.

    ``target_option`` is required for an attack and deliberately lives in
    the scenario rather than being inferred from the gold label. This keeps
    the attacker independent of ground truth; evaluation later excludes the
    accidental cases where target and gold happen to match.
    """

    attack_family: AttackFamily = AttackFamily.CLEAN
    attack_surface: AttackSurface = AttackSurface.RAG
    attack_position: AttackPosition = AttackPosition.TOP1
    target_option: Optional[str] = None
    marker: str = "PWNED"
    seed: int = 14
    defense: DefenseMode = DefenseMode.NONE
    model_track: ModelTrack = ModelTrack.API
    payload_placement: str = "prefix"
    # A template revision participates in scenario_id so --resume cannot mix
    # predictions produced under different attack text.
    payload_revision: str = "opi_medqa_r4"

    def __post_init__(self) -> None:
        if self.attack_family is not AttackFamily.CLEAN:
            if self.target_option is None or len(self.target_option.strip()) != 1:
                raise ValueError("target_option must be one option letter for attacks")
        if self.payload_placement not in {"prefix", "suffix", "sandwich"}:
            raise ValueError("payload_placement must be 'prefix', 'suffix', or 'sandwich'")
        if not self.marker.strip():
            raise ValueError("marker must be non-empty")
        if not self.payload_revision.strip():
            raise ValueError("payload_revision must be non-empty")

    @property
    def is_attack(self) -> bool:
        return self.attack_family is not AttackFamily.CLEAN

    @property
    def uses_structured_queries(self) -> bool:
        return self.defense in {DefenseMode.FRONTEND_ONLY, DefenseMode.STRUQ}

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        return {
            key: value.value if isinstance(value, Enum) else value
            for key, value in data.items()
        }

    @property
    def scenario_id(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
