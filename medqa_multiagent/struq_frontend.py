"""An API-callable StruQ-compatible structured frontend.

This module implements the *frontend* part of StruQ: a trusted instruction
channel and a separately delimited untrusted-data channel.  It intentionally
does not claim to be the official structured-instruction-tuned checkpoint;
when used with a hosted API, the provider model has not received StruQ's
training.  It is useful for an API-only, transparent structural-defense demo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from .llm_client import LLMClient, LLMResponse


_RESERVED_TOKENS = ("[INST]", "[INPT]", "[RESP]", "[MARK]", "[COLN]", "##")
_INSTRUCTION_DELIMITER = "[MARK] [INST][COLN]"
_INPUT_DELIMITER = "[MARK] [INPT][COLN]"
_RESPONSE_DELIMITER = "[MARK] [RESP][COLN]"
_PREFIX = (
    "Below is an instruction that describes a task, paired with an input "
    "that provides further context. Write a response that appropriately "
    "completes the request."
)


def filter_reserved_delimiters(text: str) -> str:
    """Remove only the reserved frontend delimiters from untrusted data."""
    filtered = text
    while True:
        updated = filtered
        for token in _RESERVED_TOKENS:
            updated = updated.replace(token, "")
        if updated == filtered:
            return updated
        filtered = updated


def compose_struq_frontend(instruction: str, untrusted_data: str) -> str:
    """Render the public StruQ-style two-channel prompt format."""
    data = filter_reserved_delimiters(untrusted_data).strip()
    return (
        f"{_PREFIX}\n\n"
        f"{_INSTRUCTION_DELIMITER}\n{instruction.strip()}\n\n"
        f"{_INPUT_DELIMITER}\n{data}\n\n"
        f"{_RESPONSE_DELIMITER}\n"
    )


@dataclass
class StruQFrontendAPIClient:
    """Wrap a hosted API client, structurally isolating known untrusted text.

    The wrapper transforms only prompts containing one of ``untrusted_parts``.
    Agent prompts without those exact strings continue to use the same direct
    API client unchanged. No result is cached or fabricated by this class.
    """

    inner: LLMClient
    untrusted_parts: Sequence[str]
    last_structured_prompt: Optional[str] = None
    structured_calls: int = 0

    def complete(
        self,
        role: str,
        prompt: str,
        model: str,
        temperature: float,
        retrieved_context_id: Optional[str] = None,
    ) -> LLMResponse:
        parts = sorted({part for part in self.untrusted_parts if part}, key=len, reverse=True)
        matched = [part for part in parts if part in prompt]
        if not matched:
            return self.inner.complete(role, prompt, model, temperature, retrieved_context_id)

        instruction = prompt
        data_sections = []
        for index, part in enumerate(matched, start=1):
            placeholder = f"<UNTRUSTED_DATA_{index}>"
            instruction = instruction.replace(part, placeholder)
            data_sections.append(f"[{placeholder}]\n{part}")
        structured_prompt = compose_struq_frontend(instruction, "\n\n".join(data_sections))
        self.last_structured_prompt = structured_prompt
        self.structured_calls += 1
        return self.inner.complete(
            role, structured_prompt, model, temperature, retrieved_context_id
        )
