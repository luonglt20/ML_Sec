"""The single black-box entrypoint: `answer_question`.

Given a question and its answer options, plus a variant selection, returns a
final answer and explanation. This is the one stable seam every later
variant (V1-V4) and the future downstream attack/defense project call
through -- `answer_question`'s signature and `AnswerResult`'s shape do not
change as later tickets add variants; only the internal behavior selected
by `variant` does.

Currently implements V0 (Direct-LLM baseline): exactly one LLM call per
question, no retrieval/agents/memory. V1-V4 raise `NotImplementedError`
until their respective tickets land.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

from .config import RunConfig
from .llm_client import LLMClient
from .parsing import parse_final_answer, strip_final_answer
from .prompts import render_direct_prompt

#: Variants implemented so far. V1-V4 are reserved names that will be added
#: by later tickets without changing this module's public signature/shape.
SUPPORTED_VARIANTS = ("V0",)


@dataclass(frozen=True)
class AnswerResult:
    """Return shape of `answer_question` -- stable across every variant.

    Attributes:
        answer: The predicted option letter, or `None` if the response was invalid.
        explanation: A short explanation of the answer (the model's free
            text, with the `Final Answer: <letter>` line itself stripped out).
        is_valid: Whether the raw response parsed to exactly one valid option letter.
        raw_response: The unparsed model text, kept for tracing.
        variant: Which variant configuration produced this result.
        prompt_tokens: Total prompt/input tokens spent answering this question.
        completion_tokens: Total completion/output tokens spent answering this question.
        total_tokens: Total tokens (prompt + completion) spent answering this question.
        latency_seconds: Total wall-clock time spent on LLM calls for this question.
        trace: Per-agent intermediate hand-off state (router query, retrieved
            passages, reasoner output, verifier decision). Empty for V0,
            since V0 has no agent hand-offs.
    """

    answer: Optional[str]
    explanation: str
    is_valid: bool
    raw_response: str
    variant: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_seconds: float
    trace: Dict[str, Any] = field(default_factory=dict)


def answer_question(
    question: str,
    options: Mapping[str, str],
    variant: str,
    config: RunConfig,
    client: LLMClient,
) -> AnswerResult:
    """Answer one MedQA-USMLE question under the given variant configuration.

    This is the single stable seam through which every variant (V0-V4) is
    invoked, and the one function the future attack/defense project
    integrates against. Its signature and `AnswerResult` return shape do not
    change for any later variant -- only the internal behavior selected by
    `variant` does.

    Args:
        question: The question stem text.
        options: Mapping of option letter to option text.
        variant: Which variant configuration to run (e.g. `"V0"`).
        config: This run's configuration (model, temperature, etc).
        client: The `LLMClient` (typically cache- and logging-wrapped) used
            for every LLM call this question requires.

    Raises:
        NotImplementedError: if `variant` isn't one of `SUPPORTED_VARIANTS` yet.
    """
    if variant not in SUPPORTED_VARIANTS:
        raise NotImplementedError(
            f"Variant {variant!r} is not yet implemented. "
            f"Implemented variants: {SUPPORTED_VARIANTS}."
        )
    return _answer_question_v0(question, options, config, client)


def _answer_question_v0(
    question: str,
    options: Mapping[str, str],
    config: RunConfig,
    client: LLMClient,
) -> AnswerResult:
    prompt = render_direct_prompt(question, options)
    response = client.complete(
        role="direct",
        prompt=prompt,
        model=config.model,
        temperature=config.temperature,
    )
    parsed = parse_final_answer(response.text)
    return AnswerResult(
        answer=parsed.answer,
        explanation=strip_final_answer(response.text),
        is_valid=parsed.is_valid,
        raw_response=response.text,
        variant="V0",
        prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens,
        total_tokens=response.total_tokens,
        latency_seconds=response.latency_seconds,
        trace={},
    )
