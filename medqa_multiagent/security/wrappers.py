"""Non-destructive attacked views over retrieval and long-term memory."""

from __future__ import annotations

import base64
import binascii
import re
import unicodedata
from dataclasses import replace
from typing import List, Mapping, Optional, Sequence

from ..memory import CaseRecord
from ..rag.retriever import Passage, Retriever
from .attacks import build_attack_payload
from .scenario import AttackPosition, AttackSurface, DefenseMode, SecurityScenario


# This guard deliberately has an interpretable, narrow policy. It operates on
# whole paragraphs so a retrieved medical reference remains intact when a
# separate injected block is removed. It is an engineering baseline, never a
# substitute label for a structured-instruction-tuned StruQ model.
UNTRUSTED_DATA_POLICY_REVISION = "guard_r2_unicode_encoded_role_boundaries"

_SUSPICIOUS_DATA_PATTERN = re.compile(
    r"(?is)(?:<\s*(?:start|end)_of_turn\s*>|\b(?:ignore|disregard|override|"
    r"execute|decode)\b.{0,160}\b(?:instruction|task|command)\b|"
    r"\b(?:final\s+answer|system\s+override|additional\s+instruction|"
    r"new\s+highest-priority)\b|\bPWNED\b|(?:^|\n)\s*(?:assistant|user|system)\s*:|"
    r"<\|(?:assistant|user|system)\|>|\[(?:INST|INPT|RESP|MARK)\])",
)

_BASE64_TOKEN = re.compile(r"(?<![A-Za-z0-9+/=])[A-Za-z0-9+/]{32,}={0,2}(?![A-Za-z0-9+/=])")
_ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d\ufeff\u2060]")


def canonicalize_untrusted_data(text: str) -> str:
    """Normalize evasive Unicode without modifying the baseline or source data."""
    normalized = unicodedata.normalize("NFKC", text).replace("\r\n", "\n")
    return _ZERO_WIDTH.sub("", normalized)


def _contains_encoded_instruction(text: str) -> bool:
    """Detect a suspicious instruction hidden in a standalone Base64 token."""
    for token in _BASE64_TOKEN.findall(text):
        try:
            decoded = base64.b64decode(token, validate=True).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError):
            continue
        if _SUSPICIOUS_DATA_PATTERN.search(canonicalize_untrusted_data(decoded)):
            return True
    return False


def is_suspicious_untrusted_data(text: str) -> bool:
    canonical = canonicalize_untrusted_data(text)
    return bool(_SUSPICIOUS_DATA_PATTERN.search(canonical) or _contains_encoded_instruction(canonical))


def filter_suspicious_untrusted_data(text: str) -> str:
    """Drop injection-shaped paragraphs from an untrusted RAG/memory field.

    The function intentionally does not infer the correct medical answer and
    does not alter passage IDs, scores, ranking, questions, or gold labels.
    It is deterministic and idempotent, which makes its clean-utility cost
    measurable in the evaluation harness.
    """
    paragraphs = re.split(r"\n\s*\n", text)
    retained = [paragraph for paragraph in paragraphs if not is_suspicious_untrusted_data(paragraph)]
    return "\n\n".join(retained).strip()


def _inject_text(text: str, payload: str, placement: str) -> str:
    if not payload:
        return text
    if placement == "prefix":
        return f"{payload}\n\n{text}"
    if placement == "suffix":
        return f"{text}\n\n{payload}"
    if placement == "sandwich":
        return f"{payload}\n\n{text}\n\n{payload}"
    raise ValueError(f"unsupported payload placement: {placement}")


def _selected_indexes(length: int, position: AttackPosition) -> Sequence[int]:
    if length <= 0:
        return ()
    if position is AttackPosition.ALL_TOP_K:
        return tuple(range(length))
    return (0,)


class InjectedRetriever:
    """Decorator that injects only into returned ``Passage.text`` values."""

    def __init__(self, retriever: Retriever, scenario: SecurityScenario) -> None:
        if scenario.attack_surface is not AttackSurface.RAG:
            raise ValueError("InjectedRetriever requires a RAG scenario")
        self._retriever = retriever
        self.scenario = scenario
        self.last_original: List[Passage] = []
        self.last_attacked: List[Passage] = []
        self.last_sanitized_ids: List[str] = []

    def retrieve(
        self,
        query: str,
        top_k: int,
        options: Optional[Mapping[str, str]] = None,
    ) -> List[Passage]:
        original = list(self._retriever.retrieve(query, top_k, options))
        self.last_original = original
        attacked = list(original)
        if self.scenario.is_attack:
            assert self.scenario.target_option is not None
            payload = build_attack_payload(
                self.scenario.attack_family,
                self.scenario.target_option,
                self.scenario.marker,
            )
            selected = set(_selected_indexes(len(original), self.scenario.attack_position))
            attacked = [
                replace(
                    passage,
                    text=_inject_text(
                        passage.text, payload, self.scenario.payload_placement
                    ),
                )
                if index in selected
                else passage
                for index, passage in enumerate(original)
            ]
        self.last_sanitized_ids = []
        if self.scenario.defense is DefenseMode.HEURISTIC_GUARD:
            guarded = []
            for passage in attacked:
                filtered = filter_suspicious_untrusted_data(passage.text)
                if filtered != passage.text:
                    self.last_sanitized_ids.append(passage.passage_id)
                    guarded.append(replace(passage, text=filtered))
                else:
                    guarded.append(passage)
            attacked = guarded
        self.last_attacked = attacked
        return attacked


class InjectedMemoryStore:
    """Read-only decorator that poisons retrieved case explanations only."""

    def __init__(self, memory_store: object, scenario: SecurityScenario) -> None:
        if scenario.attack_surface is not AttackSurface.MEMORY:
            raise ValueError("InjectedMemoryStore requires a memory scenario")
        if not hasattr(memory_store, "retrieve_exemplars"):
            raise TypeError("memory_store must implement retrieve_exemplars")
        self._memory_store = memory_store
        self.scenario = scenario
        self.last_original: List[CaseRecord] = []
        self.last_attacked: List[CaseRecord] = []
        self.last_sanitized_ids: List[str] = []

    def retrieve_exemplars(self, question: str, top_k: int = 2) -> List[CaseRecord]:
        retrieve = getattr(self._memory_store, "retrieve_exemplars")
        original = list(retrieve(question, top_k=top_k))
        self.last_original = original
        attacked = list(original)
        if self.scenario.is_attack:
            assert self.scenario.target_option is not None
            payload = build_attack_payload(
                self.scenario.attack_family,
                self.scenario.target_option,
                self.scenario.marker,
            )
            selected = set(_selected_indexes(len(original), self.scenario.attack_position))
            attacked = [
                replace(
                    case,
                    explanation=_inject_text(
                        case.explanation, payload, self.scenario.payload_placement
                    ),
                )
                if index in selected
                else case
                for index, case in enumerate(original)
            ]
        self.last_sanitized_ids = []
        if self.scenario.defense is DefenseMode.HEURISTIC_GUARD:
            guarded = []
            for case in attacked:
                filtered = filter_suspicious_untrusted_data(case.explanation)
                if filtered != case.explanation:
                    self.last_sanitized_ids.append(case.case_id)
                    guarded.append(replace(case, explanation=filtered))
                else:
                    guarded.append(case)
            attacked = guarded
        self.last_attacked = attacked
        return attacked

    def size(self) -> int:
        size = getattr(self._memory_store, "size", None)
        if callable(size):
            return int(size())
        raise AttributeError("wrapped memory store does not implement size")
