"""Loading MedQA-USMLE question pools from local JSONL files.

This module owns nothing about "dev" vs "test" splits -- it just reads a
JSONL file at a given path into a list of `Question` objects, one question
per line. Callers decide which file corresponds to which official split
(e.g. `data/dev.jsonl` vs `data/test.jsonl`, produced once by
`scripts/download_medqa.py`) and pass the loaded pool into
`medqa_multiagent.sampling.sample_dev_set` or
`medqa_multiagent.official_eval.sample_official_test_set` as appropriate --
this module has no ability to distinguish or enforce that on its own.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union


@dataclass(frozen=True)
class Question:
    """One MedQA-USMLE multiple-choice question.

    Attributes:
        question_id: Stable identifier for this question (e.g. ``"dev-00000"``).
        question: The question stem text.
        options: Mapping of option letter (e.g. ``"A"``) to option text.
        answer: The correct option letter.
    """

    question_id: str
    question: str
    options: Dict[str, str]
    answer: str


_REQUIRED_FIELDS = ("question_id", "question", "options", "answer")


def load_questions(path: Union[str, Path]) -> List[Question]:
    """Load a pool of questions from a JSONL file, one JSON object per line.

    Each line must be a JSON object with at least the fields `question_id`,
    `question`, `options`, and `answer` (see `Question`). Raises `ValueError`
    naming the offending line if a required field is missing or a line isn't
    valid JSON.
    """
    questions: List[Question] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line_number, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc

            missing = [field for field in _REQUIRED_FIELDS if field not in data]
            if missing:
                raise ValueError(
                    f"{path}:{line_number}: missing required field(s) {missing}"
                )

            questions.append(
                Question(
                    question_id=data["question_id"],
                    question=data["question"],
                    options=dict(data["options"]),
                    answer=data["answer"],
                )
            )
    return questions
