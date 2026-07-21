import sys
from pathlib import Path

# Add project root to sys.path for direct IDE execution
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import pytest
from unittest.mock import MagicMock
from medqa_multiagent.config import RunConfig


from medqa_multiagent.entrypoint import _answer_question_v2
from medqa_multiagent.rag.retriever import Passage, IndexBackedRetriever
from medqa_multiagent.rag.index import PassageRecord


from tests.fakes import FakeEmbeddingClient, FakeLLMClient, FakeVectorIndex







def test_heuristic_compression_prunes_sentences():

    embedder = FakeEmbeddingClient()
    index = FakeVectorIndex([("p1", 0.9)])
    passages_metadata = {
        "p1": PassageRecord(
            passage_id="p1", 
            source="BookA", 
            text="Ceftriaxone is a cephalosporin. It targets transpeptidases. This is unrelated filler sentence. Another useless sentence."
        )
    }

    # Case 1: Compression ON
    retriever_comp = IndexBackedRetriever(
        embedder, index, passages_metadata, retrieval_buffer=0,
        heuristic_compression=True
    )
    # We query Ceftriaxone, so only sentences containing 'Ceftriaxone' should be kept
    hits = retriever_comp.retrieve("ceftriaxone", top_k=1, options={"A": "Ceftriaxone", "B": "Gentamicin"})
    text = hits[0].text
    assert "Ceftriaxone is a cephalosporin." in text
    assert "unrelated filler" not in text  # pruned!
    assert "It targets transpeptidases." not in text  # pruned since no word matches 'ceftriaxone' or 'gentamicin'

    # Case 2: Compression OFF
    retriever_flat = IndexBackedRetriever(
        embedder, index, passages_metadata, retrieval_buffer=0,
        heuristic_compression=False
    )
    hits_flat = retriever_flat.retrieve("ceftriaxone", top_k=1)
    assert "unrelated filler" in hits_flat[0].text  # kept!


def test_option_guided_boosting():
    embedder = FakeEmbeddingClient()
    index = FakeVectorIndex([("p1", 0.5), ("p2", 0.5)])
    passages_metadata = {
        "p1": PassageRecord(passage_id="p1", source="BookA", text="This chunk is about Ceftriaxone."),
        "p2": PassageRecord(passage_id="p2", source="BookB", text="This chunk is about Gentamicin.")
    }

    mock_bm25 = MagicMock()
    mock_bm25.search.return_value = [("p1", 1.0), ("p2", 1.0)]

    retriever = IndexBackedRetriever(
        embedder, index, passages_metadata, retrieval_buffer=0,
        bm25_index=mock_bm25,
        use_option_boosting=True,
        option_boost_weight=0.1
    )

    options = {"A": "Ceftriaxone", "B": "Other"}
    hits = retriever.retrieve("query", top_k=2, options=options)

    # Ceftriaxone chunk (p1) must have higher score due to boost!
    assert hits[0].passage_id == "p1"
    assert hits[0].score > hits[1].score


def test_dynamic_top_k():
    embedder = FakeEmbeddingClient()
    index = FakeVectorIndex([("p1", 0.9), ("p2", 0.8), ("p3", 0.7)])
    passages_metadata = {
        "p1": PassageRecord(passage_id="p1", source="A", text="x"),
        "p2": PassageRecord(passage_id="p2", source="B", text="y"),
        "p3": PassageRecord(passage_id="p3", source="C", text="z"),
    }

    mock_bm25 = MagicMock()
    # Rank 0 for p1 in both dense and sparse -> RRF score = 1/60 + 1/60 = 0.0333 >= 0.030
    mock_bm25.search.return_value = [("p1", 10.0), ("p2", 5.0), ("p3", 2.0)]

    retriever = IndexBackedRetriever(
        embedder, index, passages_metadata, retrieval_buffer=0,
        bm25_index=mock_bm25,
        dynamic_top_k=True
    )

    hits = retriever.retrieve("query", top_k=3)
    assert len(hits) == 1


def test_adaptive_routing_bypasses_router():
    from medqa_multiagent.llm_client import LLMResponse

    # Setup config with Adaptive Routing ON

    config = RunConfig(
        model="fake-model",
        temperature=0.0,
        dev_sample_size=3,
        official_test_sample_size=3,
        seed=42,
        rag_top_k=2,
        rag_chunk_size=128,
        memory_top_k=1,
        rag_adaptive_routing=True,
        rag_heuristic_compression=True,
        rag_enable_backtracking=True,  # Backtracking should not run since router is bypassed
        rag_enable_debate=False
    )

    # Mock Retriever returning a High Confidence score >= 0.030 for raw question
    retriever = MagicMock()
    retriever.retrieve.return_value = [
        Passage(passage_id="p1", source="A", text="High confidence match.", score=0.035)
    ]

    # LLM Client only has responses for Reasoner & Verifier (NO Router response!)
    client = FakeLLMClient([
        "Diagnosis: C. Final Answer: C",
        "Decision: Agree\nReview Memo: Looks good."
    ])

    res = _answer_question_v2("What is ceftriaxone?", {"A": "A", "C": "C"}, config, client, retriever)

    # Asserts
    assert res.answer == "C"
    # Retriever should be called with raw question first to test confidence
    retriever.retrieve.assert_any_call("What is ceftriaxone?", 2, {"A": "A", "C": "C"})


def test_cross_encoder_rescoring_and_token_budget():

    from medqa_multiagent.rag.retriever import Passage, TokenBudgetManager, cross_encoder_rescore_passages

    passages = [
        Passage("p1", "BookA", "Generic text about medicine.", 0.5),
        Passage("p2", "BookA", "Specific text mentioning Ceftriaxone for Neisseria infection.", 0.6),
    ]

    rescored = cross_encoder_rescore_passages("Neisseria Ceftriaxone", passages)
    assert len(rescored) == 2
    assert rescored[0].passage_id == "p2"

    compressed = TokenBudgetManager.compress_passages(passages, max_total_words=6)
    assert len(compressed) >= 1
    assert "..." in compressed[1].text



def test_multi_query_retrieval_and_fusion():
    from medqa_multiagent.llm_client import LLMResponse

    config = RunConfig(

        model="fake-model",
        temperature=0.0,
        dev_sample_size=3,
        official_test_sample_size=3,
        seed=42,
        rag_top_k=2,
        rag_chunk_size=128,
        memory_top_k=1,
        rag_use_multi_query=True,
        rag_heuristic_compression=True,
        rag_adaptive_routing=False,
        rag_enable_backtracking=False,
        rag_enable_debate=True # Bật debate để nhảy vào luồng V2 nâng cao mới thay vì legacy baseline
    )

    retriever = MagicMock()
    # Mock retrieve returns
    retriever.retrieve.side_effect = lambda query, top_k, options=None: [
        Passage(passage_id=f"p_{query}", source="A", text=f"Evidence for {query}", score=0.02)
    ]

    client = FakeLLMClient([
        "Search Query 1: q1\nSearch Query 2: q2\nSearch Query 3: q3",
        "Diagnosis: C. Final Answer: C",
        "Decision: Agree\nReview Memo: Looks good."
    ])

    res = _answer_question_v2("What is ceftriaxone?", {"A": "A", "C": "C"}, config, client, retriever)

    # Asserts
    assert res.answer == "C"
    assert res.trace["router_query"] == "q1 || q2 || q3"
    # Retriever must be called for each of the 3 generated queries
    retriever.retrieve.assert_any_call("q1", 2, {"A": "A", "C": "C"})
    retriever.retrieve.assert_any_call("q2", 2, {"A": "A", "C": "C"})
    retriever.retrieve.assert_any_call("q3", 2, {"A": "A", "C": "C"})


def test_lite_context_reranker():
    embedder = FakeEmbeddingClient()
    index = FakeVectorIndex([("p1", 0.20), ("p2", 0.10)])
    passages_metadata = {
        "p1": PassageRecord(passage_id="p1", source="BookA", text="This is a simple filler chunk without any specific drugs."),
        "p2": PassageRecord(passage_id="p2", source="BookB", text="We are administering Penicillin which targets Streptococcus causing meningitis.")
    }

    # Reranker active
    retriever = IndexBackedRetriever(
        embedder, index, passages_metadata, retrieval_buffer=0,
        use_reranker=True
    )

    options = {"A": "Gentamicin", "B": "Penicillin"}
    # Query contains 'Streptococcus' and options contain 'Penicillin'
    hits = retriever.retrieve("Streptococcus treatment", top_k=2, options=options)

    # p2 must be reranked to rank 1 due to medical entity overlaps (Streptococcus, Penicillin, meningitis)
    assert hits[0].passage_id == "p2"
    assert hits[0].score > hits[1].score


def test_entity_guided_query_pruning():
    embedder = FakeEmbeddingClient()
    index = FakeVectorIndex([("p1", 0.5)])
    passages_metadata = {
        "p1": PassageRecord(passage_id="p1", source="BookA", text="Some textbook text.")
    }

    # Query pruning active
    retriever = IndexBackedRetriever(
        embedder, index, passages_metadata, retrieval_buffer=0,
        use_query_pruning=True
    )

    query = "This is a patient case. Streptococcus pneumoniae causes meningitis in adults. Which of the following is the most likely treatment?"
    retriever.retrieve(query, top_k=1)

    # Verify that the query sent to the embedding client was indeed pruned!
    # It should only contain the second sentence.
    expected_pruned_query = "Streptococcus pneumoniae causes meningitis in adults."
    assert embedder.embedded_queries[0] == expected_pruned_query


def test_medical_synonym_expansion():
    embedder = FakeEmbeddingClient()
    index = FakeVectorIndex([("p1", 0.5)])
    passages_metadata = {
        "p1": PassageRecord(passage_id="p1", source="BookA", text="Some textbook text.")
    }

    # Synonym expansion active
    retriever = IndexBackedRetriever(
        embedder, index, passages_metadata, retrieval_buffer=0,
        use_synonym_expansion=True
    )

    query = "The patient has diplococci infection."
    retriever.retrieve(query, top_k=1)

    # Verify that 'neisseria' and 'gonococcus' synonyms were appended
    assert "neisseria" in embedder.embedded_queries[0]
    assert "gonococcus" in embedder.embedded_queries[0]


def test_all_advanced_rag_opts_together():
    embedder = FakeEmbeddingClient()
    index = FakeVectorIndex([("p1", 0.20), ("p2", 0.10)])
    passages_metadata = {
        "p1": PassageRecord(passage_id="p1", source="BookA", text="This is generic text about general clinical practice."),
        "p2": PassageRecord(passage_id="p2", source="BookB", text="We must treat Neisseria infections with Gentamicin or Penicillin.")
    }

    # All advanced RAG options active
    retriever = IndexBackedRetriever(
        embedder, index, passages_metadata, retrieval_buffer=0,
        use_reranker=True,
        use_query_pruning=True,
        use_synonym_expansion=True
    )

    query = "This is a simple case. The patient has diplococci infection. Which of the following is the most likely treatment?"
    options = {"A": "Gentamicin", "B": "Penicillin"}
    hits = retriever.retrieve(query, top_k=2, options=options)

    # 1. Verify query pruning & synonym expansion
    # 'This is a simple case.' and 'Which of the following...' should be pruned.
    # 'The patient has diplococci infection.' remains, and synonyms 'neisseria', 'gonococcus' should be appended.
    pruned_expanded = embedder.embedded_queries[0]
    assert "diplococci" in pruned_expanded
    assert "neisseria" in pruned_expanded
    assert "gonococcus" in pruned_expanded
    assert "case" not in pruned_expanded
    assert "most likely" not in pruned_expanded

    # 2. Verify reranker boosted p2 to rank 1 due to entity overlaps (Neisseria, Penicillin)
    assert hits[0].passage_id == "p2"
    assert hits[0].score > hits[1].score


def test_hyde_retrieval():
    embedder = FakeEmbeddingClient()
    index = FakeVectorIndex([("p1", 0.9)])
    passages_metadata = {
        "p1": PassageRecord(passage_id="p1", source="BookA", text="Ceftriaxone is a third generation cephalosporin.")
    }

    def fake_hyde_gen(query):
        return "Hypothetical passage about Ceftriaxone treatment mechanism."

    retriever = IndexBackedRetriever(
        embedder, index, passages_metadata, retrieval_buffer=0,
        use_hyde=True, hyde_generator=fake_hyde_gen
    )

    hits = retriever.retrieve("treatment for gonorrhea", top_k=1)
    assert len(hits) == 1
    embedded_q = embedder.embedded_queries[0]
    assert "treatment for gonorrhea" in embedded_q
    assert "Hypothetical passage about Ceftriaxone" in embedded_q


def test_mmr_diversity_reranking():
    embedder = FakeEmbeddingClient()
    # p1 and p2 have identical text, p3 has distinct text
    index = FakeVectorIndex([("p1", 0.90), ("p2", 0.88), ("p3", 0.85)])
    passages_metadata = {
        "p1": PassageRecord(passage_id="p1", source="BookA", text="Neisseria gonorrhoeae causes urethritis and is treated with Ceftriaxone."),
        "p2": PassageRecord(passage_id="p2", source="BookA", text="Neisseria gonorrhoeae causes urethritis and is treated with Ceftriaxone."),
        "p3": PassageRecord(passage_id="p3", source="BookB", text="Treponema pallidum causes primary syphilis presenting as painless chancre.")
    }

    # MMR active -> p2 should be penalized due to duplicate text with p1
    retriever = IndexBackedRetriever(
        embedder, index, passages_metadata, retrieval_buffer=0,
        use_mmr=True, mmr_lambda=0.5
    )

    hits = retriever.retrieve("bacterial infection", top_k=2)
    passage_ids = [h.passage_id for h in hits]
    assert "p1" in passage_ids
    assert "p3" in passage_ids
    assert "p2" not in passage_ids






