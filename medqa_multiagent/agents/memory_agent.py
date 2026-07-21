"""Agent 3: Memory Specialist Agent module with Dedicated Toolkit and LLM support.

Queries the LongTermMemory store for past clinical case exemplars, extracts key diagnostic pearls,
and formats Memory Briefs to guide the Reasoner Agent in V3 and V4 variants.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence


from ..config import RunConfig
from ..llm_client import LLMClient
from ..memory import CaseRecord, LongTermMemory, create_default_memory_store
from ..prompts import render_memory_agent_prompt


@dataclass(frozen=True)
class MemoryAgentOutput:
    """Output state produced by the Memory Specialist Agent."""

    memory_brief: str
    exemplars: List[CaseRecord]
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_seconds: float


from .tools import create_memory_toolkit


class MemoryToolkit:
    """Dedicated Toolkit for Agent 3 (Memory Specialist Agent) with Function Calling schema."""

    def __init__(self) -> None:
        self.registry = create_memory_toolkit()

    @staticmethod
    def search_memory(
        memory_store: LongTermMemory, question: str, top_k: int
    ) -> List[CaseRecord]:
        """Tool: Search LongTermMemory store for top_k relevant past solved cases."""
        return memory_store.retrieve_exemplars(question, top_k=top_k)

    @staticmethod
    def extract_pearls(exemplars: Sequence[CaseRecord]) -> str:
        """Tool: Extract clinical pearls from exemplars."""
        if not exemplars:
            return "No previous clinical exemplars found."
        pearls = []
        for idx, ex in enumerate(exemplars, start=1):
            pearls.append(f"Exemplar [{idx}]: {ex.question[:80]}... -> Answer {ex.correct_answer}: {ex.explanation[:100]}...")
        return "\n".join(pearls)




class MemoryAgent:
    """Agent 3: Memory Specialist Agent equipped with Dedicated Toolkit and LLM Client."""

    def __init__(
        self,
        config: RunConfig,
        client: LLMClient,
        memory_store: Optional[LongTermMemory] = None,
        model: Optional[str] = None,
    ) -> None:
        self._config = config
        self._client = client
        self._memory_store = memory_store if memory_store is not None else create_default_memory_store()
        self._model = model or config.model
        self.toolkit = MemoryToolkit()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get standard function calling schemas for Memory Agent's registered tools."""
        return self.toolkit.registry.get_schemas()

    def execute_tool(self, tool_name: str, **kwargs: Any) -> Any:
        """Dynamically execute a tool from Memory Agent's toolkit by name."""
        return self.toolkit.registry.execute_tool(tool_name, **kwargs)

    def process_memory(self, question: str) -> MemoryAgentOutput:

        """Retrieve similar solved case exemplars using Memory Toolkit and format a Memory Brief."""
        exemplars = self.toolkit.search_memory(
            self._memory_store, question, top_k=self._config.memory_top_k
        )
        prompt = render_memory_agent_prompt(question, exemplars)
        response = self._client.complete(
            role="memory_specialist",
            prompt=prompt,
            model=self._model,
            temperature=self._config.temperature,
        )
        return MemoryAgentOutput(
            memory_brief=response.text,
            exemplars=exemplars,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
            latency_seconds=response.latency_seconds,
        )
