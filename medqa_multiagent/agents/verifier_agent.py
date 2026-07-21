"""Agent 5: Verifier Agent module with Dedicated Toolkit and LLM support.

Conducts Fact-Checking and Anti-Hallucination Audit on Reasoner Agent proposals,
issues Review Memos / Callback Requests, and synthesizes final Consensus decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence


from ..config import RunConfig
from ..llm_client import LLMClient
from ..parsing import parse_final_answer, strip_final_answer
from ..prompts import (
    render_verifier_final_consensus_prompt,
    render_verifier_prompt,
    render_verifier_review_prompt,
)
from ..rag.retriever import Passage


@dataclass(frozen=True)
class VerifierAuditOutput:
    """Output state produced by the Verifier Agent's peer review and fact-checking audit."""

    decision: str
    review_memo: str
    answer: Optional[str]
    explanation: str
    is_valid: bool
    raw_response: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_seconds: float


from .tools import create_verifier_toolkit, tool_anti_hallucination_fact_checker, tool_issue_callback_request


class VerifierToolkit:
    """Dedicated Toolkit for Agent 5 (Verifier Agent) with Function Calling schema."""

    def __init__(self) -> None:
        self.registry = create_verifier_toolkit()

    @staticmethod
    def audit_hallucination_risk(explanation: str, passages: Sequence[Passage]) -> bool:
        """Tool: Heuristically audit if explanation contains claims unsupported by passages."""
        result = tool_anti_hallucination_fact_checker(explanation, passages)
        return result.get("has_hallucination_risk", False)

    @staticmethod
    def issue_callback_request(review_memo: str, retry_count: int = 0) -> str:
        """Tool: Format a formal Re-retrieval Callback Request to Router Agent."""
        res = tool_issue_callback_request(review_memo, retry_count)
        return res.get("callback_message", f"Callback Request: {review_memo}")





class VerifierAgent:
    """Agent 5: Verifier / Auditor Agent equipped with Dedicated Toolkit and LLM Client."""

    def __init__(
        self,
        config: RunConfig,
        client: LLMClient,
        model: Optional[str] = None,
    ) -> None:
        self._config = config
        self._client = client
        self._model = model or config.model
        self.toolkit = VerifierToolkit()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get standard function calling schemas for Verifier Agent's registered tools."""
        return self.toolkit.registry.get_schemas()

    def execute_tool(self, tool_name: str, **kwargs: Any) -> Any:
        """Dynamically execute a tool from Verifier Agent's toolkit by name."""
        return self.toolkit.registry.execute_tool(tool_name, **kwargs)

    def audit_passages_direct(

        self,
        question: str,
        options: Mapping[str, str],
        passages: Sequence[Passage],
        candidate_answer: Optional[str],
        candidate_explanation: str,
        retrieved_context_id: Optional[str] = None,
    ) -> VerifierAuditOutput:
        """Single-pass audit against raw passages (legacy / baseline V2 path)."""
        prompt = render_verifier_prompt(
            question,
            options,
            passages,
            candidate_answer,
            candidate_explanation,
            max_passage_tokens=self._config.rag_max_passage_tokens,
        )
        response = self._client.complete(
            role="verifier",
            prompt=prompt,
            model=self._model,
            temperature=self._config.temperature,
            retrieved_context_id=retrieved_context_id,
        )
        parsed = parse_final_answer(response.text)
        explanation = strip_final_answer(response.text)
        decision = (
            "approve"
            if candidate_answer is not None and parsed.answer == candidate_answer
            else "override"
        )
        return VerifierAuditOutput(
            decision=decision,
            review_memo="",
            answer=parsed.answer,
            explanation=explanation,
            is_valid=parsed.is_valid,
            raw_response=response.text,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
            latency_seconds=response.latency_seconds,
        )

    def audit_review_brief(
        self,
        question: str,
        options: Mapping[str, str],
        research_brief: str,
        candidate_answer: Optional[str],
        candidate_explanation: str,
        retrieved_context_id: Optional[str] = None,
    ) -> VerifierAuditOutput:
        """Review candidate answer against Research Brief, detecting logical errors and hallucination."""
        prompt = render_verifier_review_prompt(
            question,
            options,
            research_brief,
            candidate_answer,
            candidate_explanation,
        )
        response = self._client.complete(
            role="verifier_review",
            prompt=prompt,
            model=self._model,
            temperature=self._config.temperature,
            retrieved_context_id=retrieved_context_id,
        )
        decision = "agree"
        review_memo = ""
        for line in response.text.splitlines():
            line_str = line.strip().lower()
            if line_str.startswith("decision:"):
                decision = line.split(":", 1)[1].strip().lower()
            elif line_str.startswith("review memo:"):
                review_memo = line.split(":", 1)[1].strip()

        parsed = parse_final_answer(response.text)
        explanation = strip_final_answer(response.text)
        return VerifierAuditOutput(
            decision=decision,
            review_memo=review_memo,
            answer=parsed.answer,
            explanation=explanation,
            is_valid=parsed.is_valid,
            raw_response=response.text,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
            latency_seconds=response.latency_seconds,
        )

    def finalize_consensus(
        self,
        question: str,
        options: Mapping[str, str],
        research_brief: str,
        reasoner_initial_proposal: Optional[str],
        review_memo: str,
        reasoner_debate_response: str,
        retrieved_context_id: Optional[str] = None,
    ) -> VerifierAuditOutput:
        """Make final executive consensus decision after evaluating the debate round."""
        prompt = render_verifier_final_consensus_prompt(
            question,
            options,
            research_brief,
            reasoner_initial_proposal,
            review_memo,
            reasoner_debate_response,
        )
        response = self._client.complete(
            role="verifier_consensus",
            prompt=prompt,
            model=self._model,
            temperature=self._config.temperature,
            retrieved_context_id=retrieved_context_id,
        )
        parsed = parse_final_answer(response.text)
        explanation = strip_final_answer(response.text)
        return VerifierAuditOutput(
            decision="consensus",
            review_memo=review_memo,
            answer=parsed.answer,
            explanation=explanation,
            is_valid=parsed.is_valid,
            raw_response=response.text,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
            latency_seconds=response.latency_seconds,
        )
