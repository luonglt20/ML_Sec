import pytest
from unittest.mock import MagicMock
from medqa_multiagent.config import RunConfig
from medqa_multiagent.entrypoint import _answer_question_v2
from tests.fakes import FakeEmbeddingClient, FakeVectorIndex




from medqa_multiagent.rag.index import PassageRecord
from medqa_multiagent.rag.retriever import IndexBackedRetriever


def test_answer_question_v2_debate_loop_trigger():
    """Verify that if Verifier disagrees, the debate loop runs and reaches consensus."""
    config = RunConfig(
        model="fake-model",
        temperature=0.0,
        dev_sample_size=3,
        official_test_sample_size=3,
        seed=42,
        rag_top_k=1,
        rag_chunk_size=128,
        memory_top_k=3,
        rag_enable_debate=True,
        rag_enable_backtracking=False,  # Keep it simple, focus on debate
    )

    passages_metadata = {
        "p1": PassageRecord(passage_id="p1", source="BookA", text="Ceftriaxone is the drug of choice for gonorrhea.")
    }
    embedder = FakeEmbeddingClient()
    index = FakeVectorIndex([("p1", 0.9)])
    retriever = IndexBackedRetriever(embedder, index, passages_metadata, retrieval_buffer=0)

    # Mock LLM Client responses
    client = MagicMock()

    # Call 1: Router query
    resp1 = MagicMock()
    resp1.text = "Search Query: ceftriaxone gonorrhea"
    resp1.prompt_tokens = 10
    resp1.completion_tokens = 5
    resp1.total_tokens = 15
    resp1.latency_seconds = 0.1

    # Call 2: Researcher Synthesis (Research Brief)
    resp2 = MagicMock()
    resp2.text = "- Clinical Findings: gonorrhea symptoms\n- Pathophysiology: biological infection\n- Treatment: Ceftriaxone"
    resp2.prompt_tokens = 20
    resp2.completion_tokens = 10
    resp2.total_tokens = 30
    resp2.latency_seconds = 0.2

    # Call 3: Reasoner propose initial (proposes incorrect answer A)
    resp3 = MagicMock()
    resp3.text = "The symptoms point to gonorrhea, but antibiotic is gentamicin.\nFinal Answer: A"
    resp3.prompt_tokens = 30
    resp3.completion_tokens = 15
    resp3.total_tokens = 45
    resp3.latency_seconds = 0.3

    # Call 4: Verifier peer reviews and DISAGREES (suggests Ceftriaxone)
    resp4 = MagicMock()
    resp4.text = "Decision: Disagree\nReview Memo: Ceftriaxone is the actual drug of choice for gonorrhea, not gentamicin. Please re-evaluate."
    resp4.prompt_tokens = 40
    resp4.completion_tokens = 20
    resp4.total_tokens = 60
    resp4.latency_seconds = 0.4

    # Call 5: Reasoner debate response (agrees and changes to C)
    resp5 = MagicMock()
    resp5.text = "You are correct, Ceftriaxone is a third-gen cephalosporin and should be used.\nFinal Answer: C"
    resp5.prompt_tokens = 50
    resp5.completion_tokens = 25
    resp5.total_tokens = 75
    resp5.latency_seconds = 0.5

    # Call 6: Verifier Final Consensus decision (re-evaluates and locks C)
    resp6 = MagicMock()
    resp6.text = "The team agrees Ceftriaxone is correct.\nFinal Answer: C"
    resp6.prompt_tokens = 60
    resp6.completion_tokens = 30
    resp6.total_tokens = 90
    resp6.latency_seconds = 0.6

    client.complete.side_effect = [resp1, resp2, resp3, resp4, resp5, resp6]

    result = _answer_question_v2(
        question="What is the drug of choice for gonorrhea?",
        options={"A": "Gentamicin", "B": "Ciprofloxacin", "C": "Ceftriaxone", "D": "Trimethoprim"},
        config=config,
        client=client,
        retriever=retriever,
    )

    # Should output final consensus answer C
    assert result.answer == "C"
    assert result.trace["decision_flow"] == "debate_consensus"
    assert len(result.trace["debate_history"]) == 1
    assert result.trace["debate_history"][0]["initial_candidate"] == "A"
    assert result.trace["debate_history"][0]["consensus_decision"] == "C"


def test_answer_question_v2_backtracking_loop():
    """Verify that if initial retrieval is insufficient, backtracking query reformulation runs."""
    config = RunConfig(
        model="fake-model",
        temperature=0.0,
        dev_sample_size=3,
        official_test_sample_size=3,
        seed=42,
        rag_top_k=1,
        rag_chunk_size=128,
        memory_top_k=3,
        rag_enable_debate=False,  # Disable debate to keep trace assertions simple
        rag_enable_backtracking=True,
        rag_max_retrieval_loops=2,
    )

    passages_metadata = {
        "p1": PassageRecord(passage_id="p1", source="BookA", text="Ceftriaxone inhibits cell wall synthesis.")
    }
    embedder = FakeEmbeddingClient()
    index = FakeVectorIndex([("p1", 0.9)])
    retriever = IndexBackedRetriever(embedder, index, passages_metadata, retrieval_buffer=0)

    client = MagicMock()

    # Call 1: Initial Router Query
    resp1 = MagicMock()
    resp1.text = "Search Query: broad query"
    resp1.prompt_tokens = 10
    resp1.completion_tokens = 5
    resp1.total_tokens = 15
    resp1.latency_seconds = 0.1

    # Call 2: Researcher checks relevance and returns INSUFFICIENT
    resp2 = MagicMock()
    resp2.text = "Relevance: Insufficient\nReason: The query 'broad query' retrieved unrelated text. We need specific cephalosporin details."
    resp2.prompt_tokens = 15
    resp2.completion_tokens = 10
    resp2.total_tokens = 25
    resp2.latency_seconds = 0.15

    # Call 3: Router reformulates query
    resp3 = MagicMock()
    resp3.text = "Search Query: Ceftriaxone cell wall"
    resp3.prompt_tokens = 20
    resp3.completion_tokens = 8
    resp3.total_tokens = 28
    resp3.latency_seconds = 0.2

    # Since rag_max_retrieval_loops = 2, after the 1st reformulation,
    # loop counter becomes 2, which fails `retrieval_loop < 2`.
    # Therefore, the while loop exits immediately without a 2nd relevance check!

    # Call 4: Researcher Synthesis (Research Brief)
    resp4 = MagicMock()
    resp4.text = "- Clinical Findings: Ceftriaxone\n- Pathophysiology: cell wall"
    resp4.prompt_tokens = 25
    resp4.completion_tokens = 12
    resp4.total_tokens = 37
    resp4.latency_seconds = 0.25

    # Call 5: Reasoner propose candidate
    resp5 = MagicMock()
    resp5.text = "Reasoner choose C.\nFinal Answer: C"
    resp5.prompt_tokens = 30
    resp5.completion_tokens = 15
    resp5.total_tokens = 45
    resp5.latency_seconds = 0.3

    # Call 6: Verifier review (Agrees immediately)
    resp6 = MagicMock()
    resp6.text = "Decision: Agree\nReview Memo: C is correct."
    resp6.prompt_tokens = 40
    resp6.completion_tokens = 20
    resp6.total_tokens = 60
    resp6.latency_seconds = 0.4

    client.complete.side_effect = [resp1, resp2, resp3, resp4, resp5, resp6]

    result = _answer_question_v2(
        question="What is Ceftriaxone's mechanism?",
        options={"A": "Gentamicin", "B": "Ciprofloxacin", "C": "Ceftriaxone", "D": "Trimethoprim"},
        config=config,
        client=client,
        retriever=retriever,
    )

    assert result.answer == "C"
    # Should record 1 loop of backtracking in history
    assert len(result.trace["backtrack_history"]) == 1
    assert result.trace["backtrack_history"][0]["relevance"] == "insufficient"
    assert result.trace["backtrack_history"][0]["query"] == "broad query"
