"""The single black-box entrypoint: `answer_question`.

Given a question and its answer options, plus a variant selection, returns a
final answer and explanation. This is the one stable seam every later
variant (V1-V4) and the future downstream attack/defense project call
through -- `answer_question`'s signature and `AnswerResult`'s shape do not
change as later tickets add variants; only the internal behavior selected
by `variant` does. The one exception is the optional, backward-compatible
`retriever` keyword argument added alongside V1: existing callers (V0,
the CLI's `answer`/`run` commands as written for #2, the demo UI) never
pass it and are entirely unaffected, since it defaults to `None` and a
real one is then built lazily, on first use, from `config.rag_index_dir`.

Currently implements V0 (Direct-LLM baseline) and V1 (RAG-only): V0 is
exactly one LLM call per question with no retrieval/agents/memory; V1 is
still exactly one LLM call, with the prompt augmented by the top-`k`
retrieved textbook passages for the question's raw text. V2-V4 raise
`NotImplementedError` until their respective tickets land.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from .config import RunConfig
from .llm_client import LLMClient
from .parsing import parse_final_answer, strip_final_answer
from .prompts import render_direct_prompt, render_rag_prompt
from .rag.retriever import Passage, Retriever

#: Variants implemented so far. V2-V4 are reserved names that will be added
#: by later tickets without changing this module's public signature/shape.
SUPPORTED_VARIANTS = ("V0", "V1")


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
    retriever: Optional[Retriever] = None,
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
        variant: Which variant configuration to run (e.g. `"V0"`, `"V1"`).
        config: This run's configuration (model, temperature, etc).
        client: The `LLMClient` (typically cache- and logging-wrapped) used
            for every LLM call this question requires.
        retriever: The `Retriever` used by variants that need retrieval
            (currently V1). Optional and backward-compatible: existing V0
            callers never pass it; when a RAG-using variant needs one and
            none was injected, a real one is built lazily from
            `config.rag_index_dir` (via `rag.client_factory.build_retriever`)
            and reused across calls. Tests inject a scripted fake instead
            (see `tests/fakes.py`), per this project's dependency-injection
            testing convention.

    Raises:
        NotImplementedError: if `variant` isn't one of `SUPPORTED_VARIANTS` yet.
    """
    if variant not in SUPPORTED_VARIANTS:
        raise NotImplementedError(
            f"Variant {variant!r} is not yet implemented. "
            f"Implemented variants: {SUPPORTED_VARIANTS}."
        )
    if variant == "V0":
        return _answer_question_v0(question, options, config, client)
    return _answer_question_v1(question, options, config, client, retriever)


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


def _compute_retrieved_context_id(passages: List[Passage]) -> str:
    """Derive a stable identifier for a retrieved-passages set.

    Used as the LLM cache key's `retrieved_context_id` component (see
    `cache_key.compute_cache_key`) -- a hash of the passage ids in their
    retrieved (best-first) order, so a differing retrieval result is a
    guaranteed cache miss even in the (extremely unlikely) case that it
    happened to render to an identical prompt string.
    """
    canonical = json.dumps([passage.passage_id for passage in passages], ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _default_retriever(config: RunConfig) -> Retriever:
    # Imported lazily so that importing this module never requires the
    # `rag` extra's heavy dependencies (faiss/transformers/torch) unless a
    # RAG-using variant is actually run without an injected retriever.
    from .rag.client_factory import build_retriever

    return build_retriever(config)


def _answer_question_v1(
    question: str,
    options: Mapping[str, str],
    config: RunConfig,
    client: LLMClient,
    retriever: Optional[Retriever],
) -> AnswerResult:
    if retriever is None:
        retriever = _default_retriever(config)

    passages = retriever.retrieve(question, config.rag_top_k)
    prompt = render_rag_prompt(question, options, passages)
    response = client.complete(
        role="rag",
        prompt=prompt,
        model=config.model,
        temperature=config.temperature,
        retrieved_context_id=_compute_retrieved_context_id(passages),
    )
    parsed = parse_final_answer(response.text)
    return AnswerResult(
        answer=parsed.answer,
        explanation=strip_final_answer(response.text),
        is_valid=parsed.is_valid,
        raw_response=response.text,
        variant="V1",
        prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens,
        total_tokens=response.total_tokens,
        latency_seconds=response.latency_seconds,
        trace={"retrieved_passages": [asdict(passage) for passage in passages]},
    )
