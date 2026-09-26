"""Durable prediction/trace record schema and JSONL reader/writer.

One record per (variant, question_id), containing at minimum: question id,
variant identifier, predicted answer, explanation, correct answer,
correctness flag, invalid-response flag, token usage, latency, and the
per-agent trace (empty for V0, since V0 has no agent hand-offs). Written as
one JSON object per line so evaluation metrics can be re-derived later
without re-calling any LLM.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union


@dataclass(frozen=True)
class PredictionRecord:
    """One (variant, question_id) prediction/trace record.

    Attributes:
        question_id: The question's stable identifier.
        variant: Which variant configuration produced this prediction.
        predicted_answer: The predicted option letter, or `None` if invalid.
        explanation: The explanation text accompanying the prediction.
        correct_answer: The question's known-correct option letter.
        is_correct: Whether `predicted_answer == correct_answer`.
        is_invalid: Whether the raw response failed to parse to a valid answer.
        prompt_tokens: Total prompt/input tokens spent answering this question.
        completion_tokens: Total completion/output tokens spent answering this question.
        total_tokens: Total tokens spent answering this question.
        latency_seconds: Total wall-clock time spent on LLM calls for this question.
        trace: Per-agent intermediate hand-off state; empty for V0.
    """

    question_id: str
    variant: str
    predicted_answer: Optional[str]
    explanation: str
    correct_answer: str
    is_correct: bool
    is_invalid: bool
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_seconds: float
    trace: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def write_prediction_records(
    records: Iterable[PredictionRecord], path: Union[str, Path]
) -> None:
    """Write `records`, one JSON object per line, to `path`.

    Creates parent directories as needed. Overwrites any existing file at
    `path` (a fresh, complete run's records replace a prior run's).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record.to_dict(), ensure_ascii=True))
            fh.write("\n")


def read_prediction_records(path: Union[str, Path]) -> List[dict]:
    """Read back a prediction/trace JSONL file into a list of plain dicts.

    Lets evaluation metrics be re-derived later purely from this file,
    without re-calling any LLM.
    """
    records: List[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
