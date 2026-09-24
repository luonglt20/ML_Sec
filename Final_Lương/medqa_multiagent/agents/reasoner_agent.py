"""Agent 4: Reasoner Agent module with Dedicated Toolkit and LLM support.

Performs step-by-step Chain-of-Thought clinical reasoning, evaluates options A-D,
cites passage IDs [1], [2] to prevent hallucination, and proposes Candidate Answers.
Also handles Debate Loop responses when challenged by the Verifier Agent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence


from ..config import RunConfig
from ..llm_client import LLMClient
from ..parsing import ParsedAnswer, parse_final_answer, strip_final_answer
from ..prompts import render_reasoner_debate_prompt, render_reasoner_prompt
from ..rag.retriever import Passage


@dataclass(frozen=True)
class ReasonerOutput:
    """Output state produced by the Reasoner Agent."""

    candidate_answer: Optional[str]
    candidate_explanation: str
    is_valid: bool
    raw_response: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_seconds: float


from .tools import create_reasoner_toolkit, tool_audit_citations


class ReasonerToolkit:
    """Dedicated Toolkit for Agent 4 (Reasoner Agent) with Function Calling schema."""

    def __init__(self) -> None:
        self.registry = create_reasoner_toolkit()

    @staticmethod
    def verify_citations(explanation: str, passage_count: int) -> bool:
        """Tool: Verify if explanation includes valid passage citations [1], [2], etc."""
        result = tool_audit_citations(explanation, passage_count)
        return result.get("has_citations", False)

    @staticmethod
    def format_option_analysis(options: Mapping[str, str]) -> str:
        """Tool: Format options list for clinical evaluation."""
        return "\n".join(f"{letter}. {text}" for letter, text in sorted(options.items()))




class ReasonerAgent:
    """Agent 4: Reasoner Agent equipped with Dedicated Toolkit and LLM Client."""

    def __init__(
        self,
        config: RunConfig,
        client: LLMClient,
        model: Optional[str] = None,
    ) -> None:
        self._config = config
        self._client = client
        self._model = model or config.model
        self.toolkit = ReasonerToolkit()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get standard function calling schemas for Reasoner Agent's registered tools."""
        return self.toolkit.registry.get_schemas()

    def execute_tool(self, tool_name: str, **kwargs: Any) -> Any:
        """Dynamically execute a tool from Reasoner Agent's toolkit by name."""
        return self.toolkit.registry.execute_tool(tool_name, **kwargs)

    def propose_candidate(

        self,
        question: str,
        options: Mapping[str, str],
        passages: Sequence[Passage],
        research_brief: Optional[str] = None,
        memory_brief: Optional[str] = None,
        retrieved_context_id: Optional[str] = None,
    ) -> ReasonerOutput:
        """Analyze clinical presentation, evaluate options A-D using Reasoner Toolkit, and propose candidate answer."""
        prompt = render_reasoner_prompt(
            question,
            options,
            passages,
            max_passage_tokens=self._config.rag_max_passage_tokens,
            research_brief=research_brief,
            memory_brief=memory_brief,
        )
        response = self._client.complete(
            role="reasoner",
            prompt=prompt,
            model=self._model,
            temperature=self._config.temperature,
            retrieved_context_id=retrieved_context_id,
        )
        parsed = parse_final_answer(response.text)
        explanation = strip_final_answer(response.text)
        return ReasonerOutput(
            candidate_answer=parsed.answer,
            candidate_explanation=explanation,
            is_valid=parsed.is_valid,
            raw_response=response.text,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
            latency_seconds=response.latency_seconds,
        )

    def debate_peer_review(
        self,
        question: str,
        options: Mapping[str, str],
        research_brief: str,
        original_explanation: str,
        review_memo: str,
        retrieved_context_id: Optional[str] = None,
    ) -> ReasonerOutput:
        """Re-evaluate clinical reasoning or defend answer when challenged by Verifier Agent in Debate Loop."""
        prompt = render_reasoner_debate_prompt(
            question,
            options,
            research_brief,
            original_explanation,
            review_memo,
        )
        response = self._client.complete(
            role="reasoner_debate",
            prompt=prompt,
            model=self._model,
            temperature=self._config.temperature,
            retrieved_context_id=retrieved_context_id,
        )
        parsed = parse_final_answer(response.text)
        explanation = strip_final_answer(response.text)
        return ReasonerOutput(
            candidate_answer=parsed.answer,
            candidate_explanation=explanation,
            is_valid=parsed.is_valid,
            raw_response=response.text,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
            latency_seconds=response.latency_seconds,
        )
