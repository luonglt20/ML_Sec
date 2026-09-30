"""Independent StruQ frontend defender for the live MedQA demo.

This module reproduces the structured-query layer from the supplied
``medqa-multiagent-defense-struq`` repository without changing the existing
semantic guard.  It is a secure-frontend defender: full StruQ additionally
requires a matched structured-instruction-tuned local checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Protocol, Sequence

from .llm_client import LLMClient, LLMResponse


SPECIAL_DELIMITER_TOKENS = ("[INST]", "[INPT]", "[RESP]", "[MARK]", "[COLN]")
FILTERED_TOKENS = SPECIAL_DELIMITER_TOKENS + ("##",)
STRUQ_INSTRUCTION_DELIMITER = "[MARK] [INST][COLN]"
STRUQ_INPUT_DELIMITER = "[MARK] [INPT][COLN]"
STRUQ_RESPONSE_DELIMITER = "[MARK] [RESP][COLN]"
STRUQ_SYSTEM_PREFIX = (
    "Below is an instruction that describes a task, paired with an input "
    "that provides further context. Write a response that appropriately "
    "completes the request."
)
STRUQ_SYSTEM_PREFIX_NO_INPUT = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request."
)


@dataclass(frozen=True)
class StructuredQuery:
    """Trusted instruction and attacker-controlled data kept in two channels."""

    instruction: str
    data: str


class StructuredCompletionClient(Protocol):
    """Capability contract from the reference StruQ completion dispatcher."""

    use_structured_queries: bool

    def complete_structured(
        self,
        role: str,
        query: StructuredQuery,
        model: str,
        temperature: float,
        retrieved_context_id: Optional[str] = None,
    ) -> LLMResponse: ...


def recursive_filter(text: str) -> str:
    """Recursively remove only StruQ's reserved tokens from untrusted data."""
    filtered = text
    while True:
        updated = filtered
        for token in FILTERED_TOKENS:
            updated = updated.replace(token, "")
        if updated == filtered:
            return updated
        filtered = updated


def compose_struq_prompt(query: StructuredQuery, *, apply_filter: bool = True) -> str:
    """Render the reference StruQ frontend format exactly."""
    data = recursive_filter(query.data) if apply_filter else query.data
    if not data.strip():
        return (
            f"{STRUQ_SYSTEM_PREFIX_NO_INPUT}\n\n"
            f"{STRUQ_INSTRUCTION_DELIMITER}\n{query.instruction.strip()}\n\n"
            f"{STRUQ_RESPONSE_DELIMITER}\n"
        )
    return (
        f"{STRUQ_SYSTEM_PREFIX}\n\n"
        f"{STRUQ_INSTRUCTION_DELIMITER}\n{query.instruction.strip()}\n\n"
        f"{STRUQ_INPUT_DELIMITER}\n{data.strip()}\n\n"
        f"{STRUQ_RESPONSE_DELIMITER}\n"
    )


def split_prompt_into_structured_query(
    prompt: str, untrusted_parts: Iterable[str]
) -> StructuredQuery:
    """Move exact untrusted fields from a legacy prompt into the data channel."""
    normalized: List[str] = []
    seen = set()
    for raw in untrusted_parts:
        part = str(raw).strip()
        if part and part not in seen:
            seen.add(part)
            normalized.append(part)
    normalized.sort(key=len, reverse=True)

    instruction = prompt
    data_sections: List[str] = []
    for part in normalized:
        if part not in instruction:
            continue
        label = f"UNTRUSTED_DATA_{len(data_sections) + 1}"
        instruction = instruction.replace(part, f"<{label}>")
        data_sections.append(f"[{label}]\n{part}")
    if not data_sections:
        raise ValueError("none of the supplied untrusted parts occur in the prompt")
    return StructuredQuery(instruction=instruction, data="\n\n".join(data_sections))


def complete_with_untrusted_data(
    client: LLMClient,
    *,
    role: str,
    prompt: str,
    model: str,
    temperature: float,
    untrusted_parts: Sequence[str],
    retrieved_context_id: Optional[str] = None,
) -> LLMResponse:
    """Use the structured channel when the supplied client advertises it."""
    complete_structured = getattr(client, "complete_structured", None)
    use_structured = getattr(client, "use_structured_queries", False) is True
    if callable(complete_structured) and use_structured and untrusted_parts:
        return complete_structured(
            role=role,
            query=split_prompt_into_structured_query(prompt, untrusted_parts),
            model=model,
            temperature=temperature,
            retrieved_context_id=retrieved_context_id,
        )
    return client.complete(
        role=role,
        prompt=prompt,
        model=model,
        temperature=temperature,
        retrieved_context_id=retrieved_context_id,
    )


@dataclass
class StruQDefender:
    """An independent LLM client decorator implementing the StruQ frontend.

    Only text identified by the caller as untrusted enters the data channel.
    Calls which do not contain such text are passed through untouched, so the
    demo never claims a structured defense was applied when it was not.
    """

    inner: LLMClient
    untrusted_parts: Sequence[str]
    use_structured_queries: bool = True
    last_query: Optional[StructuredQuery] = None
    last_structured_prompt: Optional[str] = None
    structured_calls: int = 0
    sanitization_events: int = 0

    def complete(
        self,
        role: str,
        prompt: str,
        model: str,
        temperature: float,
        retrieved_context_id: Optional[str] = None,
    ) -> LLMResponse:
        try:
            query = split_prompt_into_structured_query(prompt, self.untrusted_parts)
        except ValueError:
            return self.inner.complete(role, prompt, model, temperature, retrieved_context_id)
        return self.complete_structured(
            role, query, model, temperature, retrieved_context_id
        )

    def complete_structured(
        self,
        role: str,
        query: StructuredQuery,
        model: str,
        temperature: float,
        retrieved_context_id: Optional[str] = None,
    ) -> LLMResponse:
        self.last_query = query
        self.structured_calls += 1
        if recursive_filter(query.data) != query.data:
            self.sanitization_events += 1
        structured_prompt = compose_struq_prompt(query)
        self.last_structured_prompt = structured_prompt
        return self.inner.complete(
            role, structured_prompt, model, temperature, retrieved_context_id
        )
