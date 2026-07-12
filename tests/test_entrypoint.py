from typing import Any, Dict

import pytest

from medqa_multiagent.config import RunConfig
from medqa_multiagent.entrypoint import answer_question

from fakes import FakeLLMClient


def make_config(**overrides: Any) -> RunConfig:
    data: Dict[str, Any] = dict(
        model="gpt-4o-mini",
        temperature=0.0,
        dev_sample_size=30,
        official_test_sample_size=20,
        seed=42,
        rag_top_k=3,
        rag_chunk_size=256,
        memory_top_k=3,
    )
    data.update(overrides)
    return RunConfig(**data)


def test_v0_answer_question_returns_parsed_answer_and_explanation():
    client = FakeLLMClient(["Penicillin blocks cell wall synthesis.\nFinal Answer: C"])
    config = make_config()

    result = answer_question(
        "Which drug blocks cell wall synthesis?",
        {"A": "Gentamicin", "B": "Ciprofloxacin", "C": "Ceftriaxone", "D": "Trimethoprim"},
        "V0",
        config,
        client,
    )

    assert result.answer == "C"
    assert result.is_valid is True
    assert result.explanation == "Penicillin blocks cell wall synthesis."
    assert result.variant == "V0"
    assert result.trace == {}


def test_v0_makes_exactly_one_llm_call():
    client = FakeLLMClient(["Final Answer: A"])
    config = make_config()

    answer_question("Q?", {"A": "x", "B": "y"}, "V0", config, client)

    assert len(client.calls) == 1


def test_v0_call_uses_direct_role_and_config_model_and_temperature():
    client = FakeLLMClient(["Final Answer: A"])
    config = make_config(model="deepseek-chat", temperature=0.0)

    answer_question("Q?", {"A": "x", "B": "y"}, "V0", config, client)

    call = client.calls[0]
    assert call["role"] == "direct"
    assert call["model"] == "deepseek-chat"
    assert call["temperature"] == 0.0
    assert "Q?" in call["prompt"]
    assert "A. x" in call["prompt"]


def test_v0_invalid_response_yields_no_answer_and_is_invalid():
    client = FakeLLMClient(["I am not sure about this one."])
    config = make_config()

    result = answer_question("Q?", {"A": "x", "B": "y"}, "V0", config, client)

    assert result.answer is None
    assert result.is_valid is False


def test_v0_surfaces_token_usage_and_latency_from_the_client():
    client = FakeLLMClient(["Final Answer: A"])
    config = make_config()

    result = answer_question("Q?", {"A": "x", "B": "y"}, "V0", config, client)

    assert result.prompt_tokens == 10
    assert result.completion_tokens == 5
    assert result.total_tokens == 15
    assert result.latency_seconds == 0.01


def test_unsupported_variant_raises_not_implemented():
    client = FakeLLMClient(["Final Answer: A"])
    config = make_config()

    with pytest.raises(NotImplementedError):
        answer_question("Q?", {"A": "x", "B": "y"}, "V2", config, client)
