"""External attack/defense harness around the frozen midterm pipeline.

This module is the *only* seam used by final-project experiments.  It does
not add arguments, conditionals, or sanitization to the midterm entrypoint or
agents.  Instead it decorates an already-created retriever/memory store and
calls the unchanged baseline functions.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, Mapping, Optional

from ..config import RunConfig
from ..entrypoint import (
    AnswerResult,
    _answer_question_v4,
    _default_retriever,
    answer_question as baseline_answer_question,
)
from ..llm_client import LLMClient
from ..memory import LongTermMemory, create_default_memory_store
from ..rag.retriever import Retriever
from .scenario import AttackSurface, SecurityScenario
from .wrappers import InjectedMemoryStore, InjectedRetriever


def _security_trace(
    scenario: SecurityScenario,
    *,
    retriever: Optional[InjectedRetriever] = None,
    memory: Optional[InjectedMemoryStore] = None,
) -> Dict[str, object]:
    trace: Dict[str, object] = {
        **scenario.to_dict(),
        "scenario_id": scenario.scenario_id,
        "attack_applied": bool(retriever or memory),
    }
    if retriever is not None:
        trace["mutated_passage_ids"] = [
            attacked.passage_id
            for original, attacked in zip(retriever.last_original, retriever.last_attacked)
            if original.text != attacked.text
        ]
        trace["sanitized_passage_ids"] = list(retriever.last_sanitized_ids)
    if memory is not None:
        trace["mutated_case_ids"] = [
            attacked.case_id
            for original, attacked in zip(memory.last_original, memory.last_attacked)
            if original.explanation != attacked.explanation
        ]
        trace["sanitized_case_ids"] = list(memory.last_sanitized_ids)
    return trace


def answer_with_security_scenario(
    *,
    question: str,
    options: Mapping[str, str],
    variant: str,
    config: RunConfig,
    client: LLMClient,
    security_scenario: SecurityScenario,
    retriever: Optional[Retriever] = None,
    memory_store: Optional[LongTermMemory] = None,
) -> AnswerResult:
    """Run an attacked/guarded view without modifying the baseline package.

    RAG scenarios call the public frozen entrypoint with a decorated
    retriever.  V4 memory scenarios invoke the existing V4 baseline function
    with a decorated memory-store implementation.  The same Router,
    Researcher, Memory Agent and Reasoner code therefore executes unchanged.
    """
    if security_scenario.attack_surface is AttackSurface.RAG:
        if variant == "V0":
            raise ValueError("V0 has no retrieved RAG data to attack")
        base_retriever = retriever or _default_retriever(config)
        attacked_retriever = InjectedRetriever(base_retriever, security_scenario)
        result = baseline_answer_question(
            question, options, variant, config, client, retriever=attacked_retriever
        )
        trace = dict(result.trace)
        trace["security"] = _security_trace(
            security_scenario, retriever=attacked_retriever
        )
        return replace(result, trace=trace)

    if variant != "V4":
        raise ValueError("memory prompt-injection evaluation is scoped to V4")
    base_memory = memory_store or create_default_memory_store()
    attacked_memory = InjectedMemoryStore(base_memory, security_scenario)
    result = _answer_question_v4(
        question,
        options,
        config,
        client,
        retriever,
        attacked_memory,  # Structural typing: baseline calls retrieve_exemplars only.
    )
    trace = dict(result.trace)
    trace["security"] = _security_trace(security_scenario, memory=attacked_memory)
    return replace(result, trace=trace)
