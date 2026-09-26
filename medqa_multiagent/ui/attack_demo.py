"""Question selection helpers for the live prompt-injection UI.

This module only builds question and attack inputs. It never produces an
agent answer; the UI sends every normal, attacked, and defended input to the
configured provider API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from ..data import Question
from ..prompt_injection import choose_target_answer, inject_prompt


# Historical clean-correct -> attacked-wrong IDs from the 100-question
# undefended benchmark supplied with the related StruQ project. They only
# choose challenging inputs; the UI still calls the API for every answer.
HIGH_RISK_QUESTION_IDS = {
    "V0": ("test-00000", "test-00007", "test-00009", "test-00015", "test-00022", "test-00042", "test-00043", "test-00048"),
    "V1": ("test-00007", "test-00022", "test-00023", "test-00031", "test-00043", "test-00053", "test-00069", "test-00071"),
    "V2": ("test-00003", "test-00005", "test-00006", "test-00015", "test-00020", "test-00021", "test-00023", "test-00031"),
    "V3": ("test-00003", "test-00006", "test-00014", "test-00024", "test-00031", "test-00032", "test-00056", "test-00063"),
    "V4": ("test-00005", "test-00009", "test-00019", "test-00020", "test-00022", "test-00023", "test-00024", "test-00032"),
}


@dataclass(frozen=True)
class DemoCase:
    """One MedQA question plus the deterministic, always-wrong attack target."""

    question: Question
    target_answer: str
    attacked_question: str


def build_demo_cases(questions: Iterable[Question], variant: str = "V0") -> List[DemoCase]:
    """Return historical high-risk candidates for ``variant`` in stable order.

    ``target_answer`` is selected by ``choose_target_answer`` and therefore
    cannot equal the gold answer. Whether a provider follows that targeted
    instruction is measured live by the UI, never simulated here.
    """
    by_id = {question.question_id: question for question in questions}
    candidate_ids = HIGH_RISK_QUESTION_IDS.get(variant, HIGH_RISK_QUESTION_IDS["V0"])
    selected = [by_id[qid] for qid in candidate_ids if qid in by_id]
    if not selected:
        selected = list(by_id.values())[: len(candidate_ids)]
    return [
        DemoCase(
            question=question,
            target_answer=choose_target_answer(question.options, question.answer),
            attacked_question=inject_prompt(
                question.question,
                choose_target_answer(question.options, question.answer),
                "combine",
            ),
        )
        for question in selected
    ]
