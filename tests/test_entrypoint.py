from typing import Any, Dict

import pytest

from medqa_multiagent.config import RunConfig
from medqa_multiagent.entrypoint import answer_question
from medqa_multiagent.rag.retriever import Passage

from fakes import FakeLLMClient, FakeRetriever


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

    # V3 (not V2): V2 (multi-agent Router->Reasoner->Verifier) is now
    # implemented (see #4), so an "unsupported variant" test needs a
    # variant still awaiting its own ticket.
    with pytest.raises(NotImplementedError):
        answer_question("Q?", {"A": "x", "B": "y"}, "V3", config, client)


def _make_passages():
    return [
        Passage(passage_id="p1", source="BookA", text="Penicillin inhibits cell wall synthesis.", score=0.9),
        Passage(passage_id="p2", source="BookB", text="Beta-lactams target transpeptidases.", score=0.7),
        Passage(passage_id="p3", source="BookC", text="Resistance arises via beta-lactamases.", score=0.4),
    ]


def test_v1_answer_question_returns_parsed_answer_and_explanation():
    client = FakeLLMClient(["Beta-lactams block cell wall synthesis.\nFinal Answer: C"])
    retriever = FakeRetriever(_make_passages())
    config = make_config()

    result = answer_question(
        "Which drug blocks cell wall synthesis?",
        {"A": "Gentamicin", "B": "Ciprofloxacin", "C": "Ceftriaxone", "D": "Trimethoprim"},
        "V1",
        config,
        client,
        retriever,
    )

    assert result.answer == "C"
    assert result.is_valid is True
    assert result.explanation == "Beta-lactams block cell wall synthesis."
    assert result.variant == "V1"


def test_v1_makes_exactly_one_llm_call():
    client = FakeLLMClient(["Final Answer: A"])
    retriever = FakeRetriever(_make_passages())
    config = make_config()

    answer_question("Q?", {"A": "x", "B": "y"}, "V1", config, client, retriever)

    assert len(client.calls) == 1


def test_v1_retrieves_using_the_raw_question_text_and_configured_top_k():
    client = FakeLLMClient(["Final Answer: A"])
    retriever = FakeRetriever(_make_passages())
    config = make_config(rag_top_k=2)

    answer_question("What causes X?", {"A": "x", "B": "y"}, "V1", config, client, retriever)

    assert retriever.calls == [("What causes X?", 2)]


def test_v1_prompt_includes_retrieved_passage_text():
    client = FakeLLMClient(["Final Answer: A"])
    retriever = FakeRetriever(_make_passages())
    config = make_config(rag_top_k=3)

    answer_question("Q?", {"A": "x", "B": "y"}, "V1", config, client, retriever)

    prompt = client.calls[0]["prompt"]
    assert "Penicillin inhibits cell wall synthesis." in prompt
    assert "Beta-lactams target transpeptidases." in prompt
    assert "Resistance arises via beta-lactamases." in prompt


def test_v1_call_uses_rag_role_and_config_model_and_temperature():
    client = FakeLLMClient(["Final Answer: A"])
    retriever = FakeRetriever(_make_passages())
    config = make_config(model="deepseek-chat", temperature=0.0)

    answer_question("Q?", {"A": "x", "B": "y"}, "V1", config, client, retriever)

    call = client.calls[0]
    assert call["role"] == "rag"
    assert call["model"] == "deepseek-chat"
    assert call["temperature"] == 0.0


def test_v1_passes_a_retrieved_context_id_derived_from_the_passages():
    client = FakeLLMClient(["Final Answer: A"])
    retriever = FakeRetriever(_make_passages())
    config = make_config()

    answer_question("Q?", {"A": "x", "B": "y"}, "V1", config, client, retriever)

    assert client.calls[0]["retrieved_context_id"] is not None


def test_v1_different_retrieved_passages_yield_different_retrieved_context_ids():
    config = make_config()

    client_a = FakeLLMClient(["Final Answer: A"])
    retriever_a = FakeRetriever(_make_passages())
    answer_question("Q?", {"A": "x", "B": "y"}, "V1", config, client_a, retriever_a)

    client_b = FakeLLMClient(["Final Answer: A"])
    retriever_b = FakeRetriever(_make_passages()[:1])
    answer_question("Q?", {"A": "x", "B": "y"}, "V1", config, client_b, retriever_b)

    assert client_a.calls[0]["retrieved_context_id"] != client_b.calls[0]["retrieved_context_id"]


def test_v1_trace_includes_exactly_the_retrieved_passages():
    client = FakeLLMClient(["Final Answer: A"])
    passages = _make_passages()
    retriever = FakeRetriever(passages)
    config = make_config(rag_top_k=3)

    result = answer_question("Q?", {"A": "x", "B": "y"}, "V1", config, client, retriever)

    assert result.trace["retrieved_passages"] == [
        {"passage_id": p.passage_id, "source": p.source, "text": p.text, "score": p.score}
        for p in passages
    ]


def test_v1_invalid_response_yields_no_answer_and_is_invalid():
    client = FakeLLMClient(["I am not sure about this one."])
    retriever = FakeRetriever(_make_passages())
    config = make_config()

    result = answer_question("Q?", {"A": "x", "B": "y"}, "V1", config, client, retriever)

    assert result.answer is None
    assert result.is_valid is False


def test_v1_without_an_injected_retriever_builds_a_default_one(monkeypatch):
    client = FakeLLMClient(["Final Answer: A"])
    fake_default_retriever = FakeRetriever(_make_passages())
    monkeypatch.setattr(
        "medqa_multiagent.entrypoint._default_retriever",
        lambda config: fake_default_retriever,
    )
    config = make_config()

    result = answer_question("Q?", {"A": "x", "B": "y"}, "V1", config, client)

    assert result.variant == "V1"
    assert len(fake_default_retriever.calls) == 1


def test_v2_makes_exactly_three_llm_calls_router_reasoner_verifier():
    client = FakeLLMClient(
        [
            "cell wall synthesis inhibitors",  # router
            "Beta-lactams block cell wall synthesis.\nFinal Answer: C",  # reasoner
            "Agreed, beta-lactams are correct.\nFinal Answer: C",  # verifier
        ]
    )
    retriever = FakeRetriever(_make_passages())
    config = make_config()

    answer_question(
        "Which drug blocks cell wall synthesis?",
        {"A": "Gentamicin", "B": "Ciprofloxacin", "C": "Ceftriaxone", "D": "Trimethoprim"},
        "V2",
        config,
        client,
        retriever,
    )

    assert len(client.calls) == 3
    assert [call["role"] for call in client.calls] == ["router", "reasoner", "verifier"]


def test_v2_router_query_is_used_to_retrieve_not_the_raw_question():
    client = FakeLLMClient(
        [
            "cell wall synthesis inhibitors",
            "Final Answer: C",
            "Final Answer: C",
        ]
    )
    retriever = FakeRetriever(_make_passages())
    config = make_config(rag_top_k=3)

    answer_question(
        "Which drug blocks cell wall synthesis?", {"A": "x", "B": "y"}, "V2", config, client, retriever
    )

    assert retriever.calls == [("cell wall synthesis inhibitors", 3)]


def test_v2_verifier_approves_when_it_matches_the_reasoners_candidate():
    client = FakeLLMClient(
        [
            "query",
            "Reasoner explanation.\nFinal Answer: C",
            "Verifier agrees.\nFinal Answer: C",
        ]
    )
    retriever = FakeRetriever(_make_passages())
    config = make_config()

    result = answer_question("Q?", {"A": "x", "C": "z"}, "V2", config, client, retriever)

    assert result.answer == "C"
    assert result.explanation == "Verifier agrees."
    assert result.variant == "V2"
    assert result.trace["verifier_decision"]["decision"] == "approve"


def test_v2_verifier_overrides_when_it_disagrees_with_the_reasoners_candidate():
    client = FakeLLMClient(
        [
            "query",
            "Reasoner explanation.\nFinal Answer: A",
            "Verifier disagrees and corrects it.\nFinal Answer: C",
        ]
    )
    retriever = FakeRetriever(_make_passages())
    config = make_config()

    result = answer_question("Q?", {"A": "x", "C": "z"}, "V2", config, client, retriever)

    assert result.answer == "C"
    assert result.explanation == "Verifier disagrees and corrects it."
    assert result.trace["verifier_decision"]["decision"] == "override"
    assert result.trace["reasoner_candidate"]["answer"] == "A"


def test_v2_overrides_when_reasoner_response_is_invalid():
    client = FakeLLMClient(
        [
            "query",
            "I really don't know.",  # unparseable reasoner response
            "I'll supply my own answer.\nFinal Answer: B",
        ]
    )
    retriever = FakeRetriever(_make_passages())
    config = make_config()

    result = answer_question("Q?", {"A": "x", "B": "y"}, "V2", config, client, retriever)

    assert result.trace["reasoner_candidate"]["answer"] is None
    assert result.trace["reasoner_candidate"]["is_valid"] is False
    assert result.trace["verifier_decision"]["decision"] == "override"
    assert result.answer == "B"


def test_v2_final_answer_is_the_verifiers_not_the_reasoners_raw_candidate():
    client = FakeLLMClient(
        [
            "query",
            "Reasoner explanation.\nFinal Answer: A",
            "Verifier explanation.\nFinal Answer: B",
        ]
    )
    retriever = FakeRetriever(_make_passages())
    config = make_config()

    result = answer_question("Q?", {"A": "x", "B": "y"}, "V2", config, client, retriever)

    assert result.answer == "B"
    assert result.explanation == "Verifier explanation."


def test_v2_trace_includes_router_query_context_candidate_and_decision():
    client = FakeLLMClient(
        [
            "cell wall synthesis inhibitors",
            "Reasoner explanation.\nFinal Answer: C",
            "Verifier explanation.\nFinal Answer: C",
        ]
    )
    passages = _make_passages()
    retriever = FakeRetriever(passages)
    config = make_config()

    result = answer_question("Q?", {"A": "x", "C": "z"}, "V2", config, client, retriever)

    assert result.trace["router_query"] == "cell wall synthesis inhibitors"
    assert result.trace["retrieved_passages"] == [
        {"passage_id": p.passage_id, "source": p.source, "text": p.text, "score": p.score}
        for p in passages
    ]
    assert result.trace["reasoner_candidate"] == {
        "answer": "C",
        "explanation": "Reasoner explanation.",
        "is_valid": True,
        "raw_response": "Reasoner explanation.\nFinal Answer: C",
    }
    assert result.trace["verifier_decision"] == {
        "decision": "approve",
        "answer": "C",
        "explanation": "Verifier explanation.",
        "is_valid": True,
        "raw_response": "Verifier explanation.\nFinal Answer: C",
    }


def test_v2_sums_token_usage_and_latency_across_all_three_calls():
    client = FakeLLMClient(["query", "Final Answer: A", "Final Answer: A"])
    retriever = FakeRetriever(_make_passages())
    config = make_config()

    result = answer_question("Q?", {"A": "x", "B": "y"}, "V2", config, client, retriever)

    # FakeLLMClient scripts 10/5/15 prompt/completion/total tokens and
    # 0.01s latency per call -- three calls, so three times each.
    assert result.prompt_tokens == 30
    assert result.completion_tokens == 15
    assert result.total_tokens == 45
    assert result.latency_seconds == pytest.approx(0.03)


def test_v2_without_an_injected_retriever_builds_a_default_one(monkeypatch):
    client = FakeLLMClient(["query", "Final Answer: A", "Final Answer: A"])
    fake_default_retriever = FakeRetriever(_make_passages())
    monkeypatch.setattr(
        "medqa_multiagent.entrypoint._default_retriever",
        lambda config: fake_default_retriever,
    )
    config = make_config()

    result = answer_question("Q?", {"A": "x", "B": "y"}, "V2", config, client)

    assert result.variant == "V2"
    assert len(fake_default_retriever.calls) == 1
