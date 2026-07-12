"""Prompt rendering.

Every answer-producing prompt instructs a 1-3 sentence explanation followed
by the shared `Final Answer: <letter>` convention (see `parsing.py`), so
explanation length -- and its contribution to token cost -- stays roughly
comparable across variants. These functions return the *exact* string sent
to the model, so callers can log it verbatim.
"""

from __future__ import annotations

from typing import Mapping

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
