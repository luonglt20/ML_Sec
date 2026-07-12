"""The V0 configuration: run this run's dev sample through `answer_question`
and produce a durable prediction/trace file.

Exactly one LLM call per question, no retrieval/agents/memory -- uses #1's
parser (via `entrypoint.answer_question`) to judge validity and #1's
sampling (`sample_dev_set`) to pick the dev questions. Decoupled from the
CLI so it's independently testable and reusable (e.g. by a future
memory-building pass over the same dev sample).
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from .config import RunConfig
from .data import Question
from .entrypoint import answer_question
from .llm_client import LLMClient
from .rag.retriever import Retriever
from .records import PredictionRecord


def run_dev_evaluation(
    questions: Sequence[Question],
    variant: str,
    config: RunConfig,
    client: LLMClient,
    retriever: Optional[Retriever] = None,
) -> List[PredictionRecord]:
    """Answer every question in `questions` and return one record each.

    `questions` is expected to already be this run's sampled subset (e.g.
    the output of `medqa_multiagent.sampling.sample_dev_set`) -- this
    function itself performs no sampling.

    `retriever` is forwarded to `answer_question` for variants that need
    retrieval (currently V1); passing one built once by the caller (rather
    than leaving it `None` and letting `answer_question` lazily build a
    default per-question) avoids reloading the FAISS index/MedCPT model on
    every question in the loop.
    """
    records: List[PredictionRecord] = []
    for question in questions:
        result = answer_question(
            question.question, question.options, variant, config, client, retriever
        )
        records.append(
            PredictionRecord(
                question_id=question.question_id,
                variant=result.variant,
                predicted_answer=result.answer,
                explanation=result.explanation,
                correct_answer=question.answer,
                is_correct=result.answer == question.answer,
                is_invalid=not result.is_valid,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                total_tokens=result.total_tokens,
                latency_seconds=result.latency_seconds,
                trace=result.trace,
            )
        )
    return records
