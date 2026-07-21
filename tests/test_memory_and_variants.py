import pytest
from medqa_multiagent.config import RunConfig
from medqa_multiagent.entrypoint import answer_question, _answer_question_v3, _answer_question_v4
from medqa_multiagent.memory import CaseRecord, LongTermMemory, create_default_memory_store
from medqa_multiagent.rag.retriever import Passage
from tests.fakes import FakeLLMClient, FakeRetriever









def test_long_term_memory_retrieval():
    memory = create_default_memory_store()
    assert memory.size() == 3

    # Query about gonorrhoeae / urethral discharge
    cases = memory.retrieve_exemplars("urethral discharge and Gram-negative diplococci", top_k=1)
    assert len(cases) == 1
    assert cases[0].case_id == "case_001"
    assert "Ceftriaxone" in cases[0].explanation


def test_answer_question_v3_full_5_agent_system():
    config = RunConfig(
        model="fake-model",
        temperature=0.0,
        dev_sample_size=3,
        official_test_sample_size=3,
        seed=42,
        rag_top_k=2,
        rag_chunk_size=128,
        memory_top_k=1,
    )

    retriever = FakeRetriever([Passage("p1", "BookA", "Ceftriaxone text", 0.9)])
    # Responses for: Memory Agent -> Router -> Reasoner -> Verifier
    client = FakeLLMClient([
        "Clinical Pearl: Ceftriaxone treats Neisseria.",
        "Search Query: Neisseria treatment",
        "Diagnosis: Ceftriaxone. Final Answer: A",
        "Decision: Agree\nReview Memo: Approved. Final Answer: A",
    ])


    result = answer_question(
        "A 24-year-old male presents with urethral discharge.",
        {"A": "Ceftriaxone", "B": "Penicillin"},
        variant="V3",
        config=config,
        client=client,
        retriever=retriever,
    )

    assert result.variant == "V3"
    assert result.answer == "A"
    assert result.is_valid is True
    assert "memory_brief" in result.trace
    assert result.trace["memory_exemplars_count"] == 1


def test_answer_question_v4_ablation_without_verifier():
    config = RunConfig(
        model="fake-model",
        temperature=0.0,
        dev_sample_size=3,
        official_test_sample_size=3,
        seed=42,
        rag_top_k=2,
        rag_chunk_size=128,
        memory_top_k=1,
        rag_heuristic_compression=True,
    )

    retriever = FakeRetriever([Passage("p1", "BookA", "Ceftriaxone text", 0.9)])

    # Responses for: Memory Agent -> Router -> Reasoner (Heuristic Compression bypasses synthesis call!)
    client = FakeLLMClient([
        "Clinical Pearl: Ceftriaxone treats Neisseria.",
        "Search Query: Neisseria treatment",
        "Diagnosis: Ceftriaxone. Final Answer: A",
    ])



    result = answer_question(
        "A 24-year-old male presents with urethral discharge.",
        {"A": "Ceftriaxone", "B": "Penicillin"},
        variant="V4",
        config=config,
        client=client,
        retriever=retriever,
    )

    assert result.variant == "V4"
    assert result.answer == "A"
    assert result.is_valid is True
    assert result.trace["verifier_omitted"] is True
