"""Prompt rendering.

Every answer-producing prompt instructs a 1-3 sentence explanation followed
by the shared `Final Answer: <letter>` convention (see `parsing.py`), so
explanation length -- and its contribution to token cost -- stays roughly
comparable across variants. These functions return the *exact* string sent
to the model, so callers can log it verbatim.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

from .rag.retriever import Passage

_ANSWER_FORMAT_INSTRUCTIONS = (
    'Respond with a concise, 1-3 sentence explanation, then end your '
    'response with a line in exactly this format (no extra words on that '
    'line): "Final Answer: <letter>", where <letter> is the single letter '
    "of the option you have chosen."
)


def render_direct_prompt(question: str, options: Mapping[str, str]) -> str:
    """Render the V0 direct-LLM prompt: raw question + lettered options.

    Used verbatim as the single LLM call's prompt in the V0 (Direct-LLM
    baseline) variant -- no retrieval, no agent hand-offs.
    """
    lines = [
        "You are a medical expert answering a USMLE-style multiple-choice question.",
        "",
        "Question:",
        question.strip(),
        "",
        "Options:",
    ]
    for letter in sorted(options):
        lines.append(f"{letter}. {options[letter]}")
    lines.append("")
    lines.append(_ANSWER_FORMAT_INSTRUCTIONS)
    return "\n".join(lines)


def render_rag_prompt(
    question: str, options: Mapping[str, str], passages: Sequence[Passage]
) -> str:
    """Render the V1 RAG-augmented prompt: retrieved passages + question + options.

    Used as the single LLM call's prompt in the V1 (RAG-only) variant --
    the same one-call-per-question shape as `render_direct_prompt`, just
    with a reference-passages section prepended. `passages` is expected in
    the retriever's own ranked (best-first) order, and is rendered in that
    same order without re-sorting.
    """
    lines = [
        "You are a medical expert answering a USMLE-style multiple-choice question.",
        "",
        "Relevant reference passages:",
    ]
    if passages:
        for index, passage in enumerate(passages, start=1):
            lines.append(f"[{index}] ({passage.source}) {passage.text.strip()}")
    else:
        lines.append("(none retrieved)")
    lines.append("")
    lines.append("Question:")
    lines.append(question.strip())
    lines.append("")
    lines.append("Options:")
    for letter in sorted(options):
        lines.append(f"{letter}. {options[letter]}")
    lines.append("")
    lines.append(_ANSWER_FORMAT_INSTRUCTIONS)
    return "\n".join(lines)


def render_router_prompt(question: str) -> str:
    """Render the V2 Router agent's prompt: raw question -> retrieval query.

    Used as the Router agent's single LLM call in the V2 (and later,
    memory-augmented) multi-agent pipeline. The Router's only job is to
    reformulate the raw question stem into a concise retrieval query for
    the RAG module (`rag.retriever.Retriever`), rather than passing the
    question through verbatim the way V1 does -- so retrieval quality
    isn't limited to whatever phrasing the question happens to use.
    """
    lines = [
        "You are the Router agent in a medical question-answering pipeline.",
        "Your only job is to formulate a concise search query -- key clinical "
        "and medical terms -- that will retrieve the most relevant textbook "
        "passages for answering the question below.",
        "",
        "Question:",
        question.strip(),
        "",
        "Respond with the search query text only, on a single line. Do not "
        "answer the question itself, and do not add any explanation or "
        "extra commentary.",
    ]
    return "\n".join(lines)


def render_reasoner_prompt(
    question: str, options: Mapping[str, str], passages: Sequence[Passage]
) -> str:
    """Render the V2 Reasoner agent's prompt: retrieved context -> candidate answer.

    Used as the Reasoner agent's single LLM call in the V2 multi-agent
    pipeline. Grounded in the passages the Router agent's query retrieved
    (`render_router_prompt`), rather than a query derived from the raw
    question text itself. Produces a *candidate* answer only -- the
    Verifier agent (`render_verifier_prompt`) reviews it exactly once
    downstream, so this output is never treated as final.
    """
    lines = [
        "You are the Reasoner agent in a medical question-answering pipeline.",
        "Using the reference passages below (retrieved by a separate Router "
        "agent), propose a candidate answer to the question. A separate "
        "Verifier agent will review your candidate afterward, so focus on "
        "giving your best-grounded answer now.",
        "",
        "Relevant reference passages:",
    ]
    if passages:
        for index, passage in enumerate(passages, start=1):
            lines.append(f"[{index}] ({passage.source}) {passage.text.strip()}")
    else:
        lines.append("(none retrieved)")
    lines.append("")
    lines.append("Question:")
    lines.append(question.strip())
    lines.append("")
    lines.append("Options:")
    for letter in sorted(options):
        lines.append(f"{letter}. {options[letter]}")
    lines.append("")
    lines.append(_ANSWER_FORMAT_INSTRUCTIONS)
    return "\n".join(lines)


def render_verifier_prompt(
    question: str,
    options: Mapping[str, str],
    passages: Sequence[Passage],
    candidate_answer: Optional[str],
    candidate_explanation: str,
) -> str:
    """Render the V2 Verifier agent's prompt: review the Reasoner's candidate.

    Used as the Verifier agent's single LLM call in the V2 multi-agent
    pipeline -- reviews the Reasoner's candidate answer/explanation
    (`render_reasoner_prompt`'s output) exactly once, and either confirms
    it or overrides it with its own corrected answer/explanation. There is
    no revision loop back to the Reasoner: whatever this call outputs is
    final (see `entrypoint._answer_question_v2`). `candidate_answer` is
    `None` when the Reasoner's response didn't parse to a valid option
    letter, in which case the Verifier is shown that its input was
    unparseable rather than a fabricated letter.
    """
    lines = [
        "You are the Verifier agent in a medical question-answering pipeline.",
        "A separate Reasoner agent already proposed a candidate answer below, "
        "grounded in the reference passages also shown below. Review it "
        "exactly once: if you agree, restate the same answer letter and a "
        "brief confirming explanation; if you disagree, override it with "
        "your own corrected answer letter and explanation. There will be no "
        "further review after your response, so give your final judgment now.",
        "",
        "Relevant reference passages:",
    ]
    if passages:
        for index, passage in enumerate(passages, start=1):
            lines.append(f"[{index}] ({passage.source}) {passage.text.strip()}")
    else:
        lines.append("(none retrieved)")
    lines.append("")
    lines.append("Question:")
    lines.append(question.strip())
    lines.append("")
    lines.append("Options:")
    for letter in sorted(options):
        lines.append(f"{letter}. {options[letter]}")
    lines.append("")
    lines.append("Reasoner's candidate answer:")
    lines.append(candidate_answer if candidate_answer is not None else "(none -- invalid/unparseable response)")
    lines.append("Reasoner's candidate explanation:")
    lines.append(candidate_explanation.strip() if candidate_explanation.strip() else "(none)")
    lines.append("")
    lines.append(_ANSWER_FORMAT_INSTRUCTIONS)
    return "\n".join(lines)
