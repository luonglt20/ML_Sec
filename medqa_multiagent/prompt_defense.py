"""A narrow, deterministic input guard for the direct-question benchmark.

This is an engineering baseline inspired by CK's untrusted-text filter, not a
StruQ-trained model. It never consults answer labels or the clean question.
"""

from __future__ import annotations

import re
import unicodedata


DEFENSE_REVISION = "line_guard_r1"
_ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d\ufeff\u2060]")
_FAKE_COMPLETION = re.compile(
    r"(?i)^\s*(?:the\s+medical\s+question\s+has\s+already\s+been\s+answered|"
    r"(?:assistant|system|user)\s*:).*$"
)
_INSTRUCTION = re.compile(
    r"(?i)^\s*(?:ignore|disregard|override)\b.{0,160}\b"
    r"(?:instructions?|tasks?|commands?)\b.*$"
)
_OUTPUT_COMMAND = re.compile(
    r"(?i)^\s*(?:return|output|print|respond\s+with|say)\s+"
    r"(?:exactly\s+)?[\"']?\s*final\s+answer\s*:\s*[A-D]\b.*$"
)
_INLINE_OUTPUT_COMMAND = re.compile(
    r"(?i)\s+(?:return|output|print|respond\s+with|say)\s+"
    r"(?:exactly\s+)?[\"']?\s*final\s+answer\s*:\s*[A-D]\b.*$"
)


def guard_question(question: str) -> str:
    """Remove instruction-shaped lines/suffixes while preserving the stem.

    This deliberately targets a limited set of visible direct-injection forms.
    It cannot guarantee resistance to paraphrases or embedded instructions.
    """
    canonical = _ZERO_WIDTH.sub("", unicodedata.normalize("NFKC", question))
    retained = []
    for line in canonical.splitlines():
        if _FAKE_COMPLETION.match(line) or _INSTRUCTION.match(line) or _OUTPUT_COMMAND.match(line):
            continue
        # A one-line suffix is used by the naive attack; restrict this to a
        # command-shaped ending so clinical mentions of "final answer" remain.
        line = _INLINE_OUTPUT_COMMAND.sub("", line)
        retained.append(line)
    return "\n".join(retained).strip()
