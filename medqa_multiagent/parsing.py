"""Shared strict-regex output parser.

Every variant's raw model output is parsed by this single, pure function,
so "is this response valid" is judged identically regardless of which
variant produced it. Given raw model text, it extracts exactly one option
letter following the `Final Answer: <letter>` convention, or reports the
response as invalid.

There is deliberately no fallback re-parse attempt: this preserves genuine
variance in the required Invalid Response Rate metric across variants,
rather than letting parser leniency mask real model/architecture behavior.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional

#: Standard MedQA-USMLE option letters (4-option US split).
DEFAULT_VALID_OPTIONS = ("A", "B", "C", "D")

# Matches "Final Answer" (case-insensitive), a mandatory colon, an optional
# surrounding parenthesis, and a single letter -- provided that letter isn't
# immediately followed by another letter/digit (which would mean it's part
# of a longer, malformed token like "AB" rather than a clean single letter).
_FINAL_ANSWER_PATTERN = re.compile(
    r"Final\s+Answer\s*:\s*\(?\s*([A-Za-z])\s*\)?(?![A-Za-z0-9])",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ParsedAnswer:
    """Result of parsing one raw model response.

    Attributes:
        is_valid: Whether exactly one valid option letter was extracted.
        answer: The extracted, uppercased option letter, or `None` if invalid.
        raw_text: The original raw text that was parsed (kept for tracing).
        reason: A short, human-readable explanation of why parsing failed;
            `None` when `is_valid` is `True`.
    """

    is_valid: bool
    answer: Optional[str]
    raw_text: str
    reason: Optional[str] = None


def parse_final_answer(
    raw_text: str, valid_options: Iterable[str] = DEFAULT_VALID_OPTIONS
) -> ParsedAnswer:
    """Extract the option letter from `raw_text`'s "Final Answer: <letter>" line.

    Returns a `ParsedAnswer` with `is_valid=True` and the (uppercased)
    answer letter only when `raw_text` contains exactly one occurrence of
    the convention, and its letter is one of `valid_options`. Every other
    case -- no occurrence, more than one occurrence (even if the letters
    agree), or a letter outside `valid_options` -- is reported invalid.
    There is no fallback re-parse attempt.
    """
    valid_set = {option.upper() for option in valid_options}
    matches = [m.group(1).upper() for m in _FINAL_ANSWER_PATTERN.finditer(raw_text)]

    if len(matches) == 0:
        return ParsedAnswer(
            is_valid=False,
            answer=None,
            raw_text=raw_text,
            reason="no 'Final Answer: <letter>' pattern found",
        )

    if len(matches) > 1:
        return ParsedAnswer(
            is_valid=False,
            answer=None,
            raw_text=raw_text,
            reason=f"ambiguous: found {len(matches)} 'Final Answer:' occurrences",
        )

    letter = matches[0]
    if letter not in valid_set:
        return ParsedAnswer(
            is_valid=False,
            answer=None,
            raw_text=raw_text,
            reason=f"'{letter}' is not one of the valid options {sorted(valid_set)}",
        )

    return ParsedAnswer(is_valid=True, answer=letter, raw_text=raw_text)
