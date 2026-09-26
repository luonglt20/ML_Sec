"""Agent 1: Router Agent module with Dedicated Toolkit and LLM support.

Classifies question types, extracts medical entities, and formulates single or multi-query search strategies
including HyDE (Hypothetical Document Embeddings) and backtracking reformulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Set


from ..config import RunConfig
from ..llm_client import LLMClient
from ..prompts import (
    render_multi_query_prompt,
    render_router_prompt,
    render_router_reformulate_prompt,
)


@dataclass(frozen=True)
class RouterOutput:
    """Output state produced by the Router Agent."""

    query: str
    raw_response: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_seconds: float
    entities: Set[str]


def parse_router_query(router_text: str) -> str:
    """Extract the ``Search Query`` from a structured Router agent response."""
    for line in router_text.splitlines():
        if line.strip().lower().startswith("search query:"):
            return line.split(":", 1)[1].strip()
    return router_text.strip()


from .tools import (
    create_router_toolkit,
    tool_decompose_subqueries,
    tool_extract_medical_entities,
    tool_fast_thinking_evaluator,
    tool_generate_hyde_context,
    tool_prune_vignette_noise,
    tool_search_strategy_planner,
)


class RouterToolkit:
    """Dedicated Toolkit for Agent 1 (Router Agent) with Function Calling schema (6 Tools)."""

    def __init__(self) -> None:
        self.registry = create_router_toolkit()

    @staticmethod
    def extract_entities(text: str) -> Set[str]:
        """Tool 1: Extract capitalized acronyms and medical entity keywords from query."""
        return tool_extract_medical_entities(text)

    @staticmethod
    def generate_hyde_context(question: str) -> str:
        """Tool 2: Generate a hypothetical textbook passage context snippet."""
        return tool_generate_hyde_context(question)

    @staticmethod
    def decompose_subqueries(question: str) -> List[str]:
        """Tool 3: Heuristically decompose clinical question into 3 sub-query focuses."""
        return tool_decompose_subqueries(question)

    @staticmethod
    def prune_vignette_noise(question: str) -> str:
        """Tool 4: Remove sentence fluff and question prompt noise from clinical vignette."""
        return tool_prune_vignette_noise(question)

    @staticmethod
    def fast_thinking_evaluator(question: str, options: Mapping[str, str]) -> Dict[str, Any]:
        """Tool 5: Fast Thinking Evaluator - Detect simple questions for fast-path processing."""
        return tool_fast_thinking_evaluator(question, options)

    @staticmethod
    def search_strategy_planner(question: str) -> Dict[str, Any]:
        """Tool 6: Plan search strategy (Dense vs Sparse vs Hybrid RAG)."""
        return tool_search_strategy_planner(question)





class RouterAgent:
    """Agent 1: Router / Triage Agent equipped with Dedicated Toolkit and LLM Client."""

    def __init__(
        self,
        config: RunConfig,
        client: LLMClient,
        model: Optional[str] = None,
    ) -> None:
        self._config = config
        self._client = client
        self._model = model or config.model
        self.toolkit = RouterToolkit()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get standard function calling schemas for Router Agent's registered tools."""
        return self.toolkit.registry.get_schemas()

    def execute_tool(self, tool_name: str, **kwargs: Any) -> Any:
        """Dynamically execute a tool from Router Agent's toolkit by name."""
        return self.toolkit.registry.execute_tool(tool_name, **kwargs)

    def formulate_query(self, question: str) -> RouterOutput:

        """Formulate a structured single search query using Router Toolkit."""
        prompt = render_router_prompt(question)
        response = self._client.complete(
            role="router",
            prompt=prompt,
            model=self._model,
            temperature=self._config.temperature,
        )
        query = parse_router_query(response.text)
        entities = self.toolkit.extract_entities(question)
        return RouterOutput(
            query=query,
            raw_response=response.text,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
            latency_seconds=response.latency_seconds,
            entities=entities,
        )

    def formulate_multi_queries(self, question: str) -> RouterOutput:
        """Formulate 3 distinct search queries using Router Toolkit."""
        prompt = render_multi_query_prompt(question)
        response = self._client.complete(
            role="router_multi",
            prompt=prompt,
            model=self._model,
            temperature=self._config.temperature,
        )
        queries = []
        for line in response.text.splitlines():
            if "search query" in line.lower():
                parts = line.split(":", 1)
                if len(parts) > 1:
                    queries.append(parts[1].strip())

        if not queries:
            queries = self.toolkit.decompose_subqueries(question)

        query_str = " || ".join(queries)
        entities = self.toolkit.extract_entities(question)
        return RouterOutput(
            query=query_str,
            raw_response=response.text,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
            latency_seconds=response.latency_seconds,
            entities=entities,
        )

    def reformulate_query(
        self,
        question: str,
        failed_query: str,
        feedback_reason: str,
    ) -> RouterOutput:
        """Reformulate search query based on feedback in Backtracking Loop."""
        prompt = render_router_reformulate_prompt(question, failed_query, feedback_reason)
        response = self._client.complete(
            role="router_reformulate",
            prompt=prompt,
            model=self._model,
            temperature=self._config.temperature,
        )
        query = parse_router_query(response.text)
        entities = self.toolkit.extract_entities(question)
        return RouterOutput(
            query=query,
            raw_response=response.text,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
            latency_seconds=response.latency_seconds,
            entities=entities,
        )
