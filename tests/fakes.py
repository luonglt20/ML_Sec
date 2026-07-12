"""Deterministic test doubles for the LLM client interface.

Per this project's Testing Decisions, the LLM client is the dependency-
injection point tests substitute with scripted fakes rather than mocking
any agent internals.
"""

from __future__ import annotations

from typing import List, Optional

from medqa_multiagent.llm_client import LLMResponse


class FakeLLMClient:
    """Returns scripted response texts in order, one per `complete` call.

    Records every call's kwargs in `self.calls`, so tests can assert on how
    many times (and with what parameters) the client was actually invoked --
    e.g. to prove a cache hit avoided a call, or that the Verifier's
    single-pass behavior doesn't trigger a second Reasoner call.
    """

    def __init__(self, responses: List[str]) -> None:
        self._responses = list(responses)
        self.calls: List[dict] = []

    def complete(
        self,
        role: str,
        prompt: str,
        model: str,
        temperature: float,
        retrieved_context_id: Optional[str] = None,
    ) -> LLMResponse:
        self.calls.append(
            {
                "role": role,
                "prompt": prompt,
                "model": model,
                "temperature": temperature,
                "retrieved_context_id": retrieved_context_id,
            }
        )
        if not self._responses:
            raise AssertionError("FakeLLMClient ran out of scripted responses")
        text = self._responses.pop(0)
        return LLMResponse(
            text=text,
            model=model,
            temperature=temperature,
            prompt=prompt,
            role=role,
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            latency_seconds=0.01,
            retrieved_context_id=retrieved_context_id,
        )
