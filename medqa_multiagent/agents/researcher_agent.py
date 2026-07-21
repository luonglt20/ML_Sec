"""Agent 2: Researcher Agent module with Dedicated Toolkit and LLM support.

Executes RAG retrieval (Dense FAISS + Sparse BM25 + MMR Diversity + HyDE), evaluates relevance sufficiency,
handles backtracking re-retrieval loops, and synthesizes retrieved passages into a structured Research Brief.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set


from ..config import RunConfig
from ..llm_client import LLMClient
from ..prompts import (
    render_researcher_feedback_prompt,
    render_researcher_synthesis_prompt,
)
from ..rag.retriever import Passage, Retriever


@dataclass(frozen=True)
class ResearcherEvalOutput:
    """Output evaluation from the Researcher Agent checking retrieval sufficiency."""

    relevance: str
    reason: str
    suggested_focus: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_seconds: float


@dataclass(frozen=True)
class ResearcherSynthesisOutput:
    """Synthesized Research Brief output from the Researcher Agent."""

    research_brief: str
    passages: List[Passage]
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_seconds: float


from .tools import create_researcher_toolkit, tool_evaluate_passages_coverage, tool_mmr_filter_passages


class ResearcherToolkit:
    """Dedicated Toolkit for Agent 2 (Researcher Agent) with Function Calling schema."""

    def __init__(self) -> None:
        self.registry = create_researcher_toolkit()

    @staticmethod
    def mmr_filter(passages: List[Passage], top_k: int, mmr_lambda: float = 0.7) -> List[Passage]:
        """Tool: Apply Maximal Marginal Relevance (MMR) filtering to reduce passage redundancy."""
        return tool_mmr_filter_passages(passages, top_k, mmr_lambda)

    @staticmethod
    def evaluate_passages_coverage(question: str, passages: Sequence[Passage]) -> bool:
        """Tool: Heuristically check if passages contain essential clinical keywords."""
        result = tool_evaluate_passages_coverage(question, passages)
        return result.get("is_sufficient", False)




class ResearcherAgent:
    """Agent 2: Researcher / Retrieval Agent equipped with Dedicated Toolkit and LLM Client."""

    def __init__(
        self,
        config: RunConfig,
        client: LLMClient,
        retriever: Retriever,
        model: Optional[str] = None,
    ) -> None:
        self._config = config
        self._client = client
        self._retriever = retriever
        self._model = model or config.model
        self.toolkit = ResearcherToolkit()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get standard function calling schemas for Researcher Agent's registered tools."""
        return self.toolkit.registry.get_schemas()

    def execute_tool(self, tool_name: str, **kwargs: Any) -> Any:
        """Dynamically execute a tool from Researcher Agent's toolkit by name."""
        return self.toolkit.registry.execute_tool(tool_name, **kwargs)

    def retrieve(

        self,
        query: str,
        options: Optional[Mapping[str, str]] = None,
    ) -> List[Passage]:
        """Execute RAG retrieval and MMR filtering using Researcher Toolkit."""
        if "||" in query:
            sub_queries = [q.strip() for q in query.split("||") if q.strip()]
            all_passages: List[Passage] = []
            for sq in sub_queries:
                all_passages.extend(self._retriever.retrieve(sq, self._config.rag_top_k, options))
            seen_ids = set()
            unique_passages = []
            for p in sorted(all_passages, key=lambda x: -x.score):
                if p.passage_id not in seen_ids:
                    seen_ids.add(p.passage_id)
                    unique_passages.append(p)
            raw_results = unique_passages[: self._config.rag_top_k * 2]
        else:
            raw_results = self._retriever.retrieve(query, self._config.rag_top_k * 2, options)

        if getattr(self._config, "rag_use_mmr", False):
            return self.toolkit.mmr_filter(
                raw_results,
                self._config.rag_top_k,
                getattr(self._config, "rag_mmr_lambda", 0.7),
            )
        return raw_results[: self._config.rag_top_k]

    def evaluate_relevance(
        self,
        question: str,
        query: str,
        passages: Sequence[Passage],
    ) -> ResearcherEvalOutput:
        """Evaluate if retrieved passages are sufficient using Researcher Toolkit & LLM."""
        prompt = render_researcher_feedback_prompt(question, query, passages)
        response = self._client.complete(
            role="researcher_eval",
            prompt=prompt,
            model=self._model,
            temperature=self._config.temperature,
        )

        relevance = "sufficient"
        reason = ""
        suggested_focus = ""
        for line in response.text.splitlines():
            line_str = line.strip().lower()
            if line_str.startswith("relevance:"):
                relevance = line.split(":", 1)[1].strip().lower()
            elif line_str.startswith("reason:"):
                reason = line.split(":", 1)[1].strip()
            elif line_str.startswith("suggested focus:"):
                suggested_focus = line.split(":", 1)[1].strip()

        return ResearcherEvalOutput(
            relevance=relevance,
            reason=reason,
            suggested_focus=suggested_focus,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
            latency_seconds=response.latency_seconds,
        )

    def synthesize_brief(
        self,
        question: str,
        passages: Sequence[Passage],
    ) -> ResearcherSynthesisOutput:
        """Synthesize retrieved raw passages into a Research Brief."""
        if not passages:
            return ResearcherSynthesisOutput(
                research_brief="No clinical evidence retrieved.",
                passages=[],
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                latency_seconds=0.0,
            )

        if self._config.rag_heuristic_compression:
            lines = ["Clinical Evidence (Heuristically Compressed):"]
            for index, passage in enumerate(passages, start=1):
                lines.append(f"[{index}] ({passage.source}) {passage.text.strip()}")
            brief = "\n".join(lines)
            return ResearcherSynthesisOutput(
                research_brief=brief,
                passages=list(passages),
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                latency_seconds=0.0,
            )

        prompt = render_researcher_synthesis_prompt(question, passages)
        response = self._client.complete(
            role="researcher_synthesis",
            prompt=prompt,
            model=self._model,
            temperature=self._config.temperature,
        )
        return ResearcherSynthesisOutput(
            research_brief=response.text,
            passages=list(passages),
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
            latency_seconds=response.latency_seconds,
        )
