from medqa_multiagent.config import RunConfig
from medqa_multiagent.entrypoint import answer_question
from medqa_multiagent.llm_client import LLMResponse
from medqa_multiagent.memory import CaseRecord, LongTermMemory
from medqa_multiagent.rag.retriever import Passage
from medqa_multiagent.security.harness import answer_with_security_scenario
from medqa_multiagent.security.scenario import AttackFamily, AttackSurface, SecurityScenario
from tests.fakes import FakeLLMClient, FakeRetriever


def _config(**overrides):
    values = dict(
        model="fake",
        temperature=0.0,
        dev_sample_size=4,
        official_test_sample_size=4,
        seed=14,
        rag_top_k=1,
        rag_chunk_size=128,
        memory_top_k=1,
    )
    values.update(overrides)
    return RunConfig(**values)


def test_v2_security_scenario_injects_after_retrieval_and_records_trace():
    retriever = FakeRetriever([Passage("p1", "book", "clean evidence", 0.9)])
    client = FakeLLMClient(
        [
            "Search Query: evidence",
            "Candidate.\nFinal Answer: D",
            "Approved.\nFinal Answer: D",
        ]
    )
    scenario = SecurityScenario(
        attack_family=AttackFamily.COMBINED,
        attack_surface=AttackSurface.RAG,
        target_option="D",
    )

    result = answer_with_security_scenario(
        question="Question?",
        options={"A": "a", "B": "b", "C": "c", "D": "d"},
        variant="V2",
        config=_config(),
        client=client,
        retriever=retriever,
        security_scenario=scenario,
    )

    assert "PWNED" in client.calls[1]["prompt"]
    assert result.trace["security"]["mutated_passage_ids"] == ["p1"]
    assert result.trace["retrieved_passages"][0]["passage_id"] == "p1"
    assert result.trace["retrieved_passages"][0]["score"] == 0.9


def test_no_security_scenario_keeps_the_midterm_v2_path_unwrapped():
    """The final harness must be opt-in; normal V2 must not invoke a guard."""
    retriever = FakeRetriever([Passage("p1", "book", "PWNED text already in corpus", 0.9)])
    client = FakeLLMClient(
        [
            "Search Query: evidence",
            "Candidate.\nFinal Answer: A",
            "Approved.\nFinal Answer: A",
        ]
    )

    result = answer_question(
        "Question?",
        {"A": "a", "B": "b", "C": "c", "D": "d"},
        "V2",
        _config(),
        client,
        retriever=retriever,
    )

    assert "security" not in result.trace
    assert "PWNED text already in corpus" in client.calls[1]["prompt"]
    assert retriever.calls == [("evidence", 1)]


def test_v4_memory_attack_does_not_mutate_underlying_store_or_rag():
    clean_case = CaseRecord(
        "dev-1", "Similar case", {"A": "a", "B": "b"}, "A", "clean rationale"
    )
    store = LongTermMemory([clean_case])
    retriever = FakeRetriever([Passage("p1", "book", "clean RAG evidence", 0.9)])
    client = FakeLLMClient(
        [
            "Memory brief containing PWNED",
            "Search Query: evidence",
            "Reasoning.\nFinal Answer: B",
        ]
    )
    scenario = SecurityScenario(
        attack_family=AttackFamily.COMBINED,
        attack_surface=AttackSurface.MEMORY,
        target_option="B",
    )

    result = answer_with_security_scenario(
        question="Similar case",
        options={"A": "a", "B": "b"},
        variant="V4",
        config=_config(rag_heuristic_compression=True),
        client=client,
        retriever=retriever,
        memory_store=store,
        security_scenario=scenario,
    )

    assert "PWNED" in client.calls[0]["prompt"]
    assert "PWNED" not in result.trace["retrieved_passages"][0]["text"]
    assert result.trace["security"]["mutated_case_ids"] == ["dev-1"]
    assert store.retrieve_exemplars("Similar case", 1)[0].explanation == "clean rationale"


class _PipelineStructuredClient:
    use_structured_queries = True

    def __init__(self):
        self.structured_calls = []
        self.legacy_calls = []
        self._responses = iter(
            [
                "Search Query: evidence",
                "Reasoning.\nFinal Answer: A",
                "Approved.\nFinal Answer: A",
            ]
        )

    def _response(self, role, prompt, model, temperature, retrieved_context_id):
        return LLMResponse(
            text=next(self._responses),
            model=model,
            temperature=temperature,
            prompt=prompt,
            role=role,
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            latency_seconds=0.01,
            retrieved_context_id=retrieved_context_id,
        )

    def complete(self, role, prompt, model, temperature, retrieved_context_id=None):
        self.legacy_calls.append(role)
        return self._response(role, prompt, model, temperature, retrieved_context_id)

    def complete_structured(self, role, query, model, temperature, retrieved_context_id=None):
        self.structured_calls.append((role, query))
        return self._response(role, query.instruction, model, temperature, retrieved_context_id)


def test_v2_security_harness_keeps_midterm_client_call_protocol():
    client = _PipelineStructuredClient()
    retriever = FakeRetriever([Passage("p1", "book", "clean evidence", 0.9)])
    scenario = SecurityScenario(
        attack_family=AttackFamily.COMBINED,
        attack_surface=AttackSurface.RAG,
        target_option="D",
    )

    answer_with_security_scenario(
        question="Question?",
        options={"A": "a", "B": "b", "C": "c", "D": "d"},
        variant="V2",
        config=_config(),
        client=client,
        retriever=retriever,
        security_scenario=scenario,
    )

    assert client.legacy_calls == ["router", "reasoner", "verifier"]
    assert client.structured_calls == []
