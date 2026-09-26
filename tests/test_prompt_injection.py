import pytest

from medqa_multiagent.config import RunConfig
from medqa_multiagent.data import Question
from medqa_multiagent.prompt_injection import (
    ATTACK_STRATEGIES,
    evaluate_prompt_injection,
    guard_prompt_injection_input,
    inject_prompt,
)

from tests.fakes import FakeLLMClient


@pytest.mark.parametrize("strategy", ATTACK_STRATEGIES)
def test_input_guard_removes_each_benchmark_attack_suffix(strategy):
    question = "Which treatment is indicated?"
    attacked = inject_prompt(question, "C", strategy)

    guarded, sanitized = guard_prompt_injection_input(attacked)

    assert guarded == question
    assert sanitized is True


def test_input_guard_leaves_clean_question_unchanged():
    question = "Which treatment is indicated?"

    guarded, sanitized = guard_prompt_injection_input(question)

    assert guarded == question
    assert sanitized is False


def test_evaluation_applies_input_guard_to_all_answer_calls():
    question = Question(
        question_id="q1",
        question="Which treatment is indicated?",
        options={"A": "one", "B": "two", "C": "three", "D": "four"},
        answer="A",
    )
    client = FakeLLMClient(["Final Answer: A", "Final Answer: A"])

    records, metrics = evaluate_prompt_injection(
        [question],
        "V0",
        "combine",
        RunConfig(model="test-model", temperature=0.0),
        client,
        max_workers=1,
        use_input_guard=True,
    )

    assert records[0].clean_input_sanitized is False
    assert records[0].attacked_input_sanitized is True
    assert "Ignore previous instructions" not in client.calls[1]["prompt"]
    assert metrics.attacked_input_sanitization_rate == 1.0
