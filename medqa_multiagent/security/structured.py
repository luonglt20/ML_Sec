"""StruQ-compatible structured-query frontend and completion dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Protocol, Sequence

from ..llm_client import LLMClient, LLMResponse


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
    """A trusted task instruction separated from attacker-controlled data."""

    instruction: str
    data: str


class StructuredCompletionClient(Protocol):
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
    """Remove StruQ-reserved delimiter strings from untrusted data.

    The loop mirrors the official frontend and is deliberately case-sensitive,
    matching the pretrained model's reserved special-token vocabulary.
    """
    filtered = text
    while True:
        updated = filtered
        for token in FILTERED_TOKENS:
            updated = updated.replace(token, "")
        if updated == filtered:
            return updated
        filtered = updated


def compose_struq_prompt(query: StructuredQuery, *, apply_filter: bool = True) -> str:
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
    prompt: str,
    untrusted_parts: Iterable[str],
) -> StructuredQuery:
    """Replace exact untrusted chunks with placeholders in a legacy prompt.

    Callers pass the exact rendered passage/brief/explanation strings. Sorting
    longest-first prevents a short chunk from corrupting a longer replacement.
    Parts absent from the prompt are ignored: some branches generate a prompt
    from a synthesized brief instead of the underlying passages.
    """
    normalized: List[str] = []
    seen = set()
    for raw in untrusted_parts:
        part = str(raw).strip()
        if not part or part in seen:
            continue
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
        # A structured query with no data would give a false sense of
        # separation. Fail closed for clients explicitly configured as StruQ.
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
    """Use two-channel completion when supported; otherwise preserve legacy behavior."""
    # ``is True`` prevents permissive mocks (which synthesize arbitrary
    # attributes) from accidentally entering the structured path.
    complete_structured = getattr(client, "complete_structured", None)
    use_structured = getattr(client, "use_structured_queries", False) is True
    if callable(complete_structured) and use_structured and untrusted_parts:
        query = split_prompt_into_structured_query(prompt, untrusted_parts)
        return complete_structured(
            role=role,
            query=query,
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
