"""Deterministic MedQA sampling and train/dev-only memory construction."""

from __future__ import annotations

import random
from typing import Dict, List, Sequence

from ..data import Question
from ..memory import CaseRecord, LongTermMemory


def stratified_question_sample(
    questions: Sequence[Question],
    *,
    total: int = 100,
    seed: int = 14,
) -> List[Question]:
    """Sample equal counts per answer label, then return stable ID order."""
    if total <= 0:
        raise ValueError("total must be positive")
    labels = sorted({question.answer.upper() for question in questions})
    if not labels or total % len(labels) != 0:
        raise ValueError("total must be divisible by the number of answer labels")
    per_label = total // len(labels)
    grouped: Dict[str, List[Question]] = {label: [] for label in labels}
    for question in questions:
        grouped[question.answer.upper()].append(question)

    rng = random.Random(seed)
    selected: List[Question] = []
    for label in labels:
        candidates = sorted(grouped[label], key=lambda item: item.question_id)
        if len(candidates) < per_label:
            raise ValueError(f"not enough questions for answer label {label}")
        selected.extend(rng.sample(candidates, per_label))
    return sorted(selected, key=lambda item: item.question_id)


def memory_from_dev_questions(questions: Sequence[Question]) -> LongTermMemory:
    """Build a leakage-free memory store from the dev split.

    MedQA JSONL does not contain rationales, so the explanation records only
    provenance and the known option. It is sufficient for testing whether an
    instruction embedded in a retrieved memory record crosses the trust boundary.
    """
    cases = [
        CaseRecord(
            case_id=question.question_id,
            question=question.question,
            options=dict(question.options),
            correct_answer=question.answer,
            explanation=(
                "This case was solved in the MedQA development split; the "
                f"recorded correct option is {question.answer}."
            ),
        )
        for question in questions
    ]
    return LongTermMemory(cases)
