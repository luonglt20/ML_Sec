from medqa_multiagent.rag.index import PassageRecord
from medqa_multiagent.rag.retriever import IndexBackedRetriever

from tests.fakes import FakeEmbeddingClient, FakeVectorIndex



def make_passages():
    return {
        "p1": PassageRecord(passage_id="p1", source="BookA", text="first passage"),
        "p2": PassageRecord(passage_id="p2", source="BookB", text="second passage"),
        "p3": PassageRecord(passage_id="p3", source="BookC", text="third passage"),
    }


def test_retrieve_embeds_the_query_and_searches_the_index():
    embedder = FakeEmbeddingClient({"what causes X?": [1.0, 2.0]})
    index = FakeVectorIndex([("p1", 0.9), ("p2", 0.5)])
    # retrieval_buffer=0 so fetch_k == top_k exactly (keeps assertion simple)
    retriever = IndexBackedRetriever(embedder, index, make_passages(), retrieval_buffer=0)

    passages = retriever.retrieve("what causes X?", top_k=2)

    assert embedder.embedded_queries == ["what causes X?"]
    assert index.searched_vectors == [[1.0, 2.0]]
    assert index.searched_top_k == [2]  # fetch_k = top_k + buffer = 2+0 = 2
    assert [p.passage_id for p in passages] == ["p1", "p2"]


def test_retrieve_resolves_passage_metadata_and_score_in_ranked_order():
    embedder = FakeEmbeddingClient()
    index = FakeVectorIndex([("p2", 0.95), ("p3", 0.42)])
    retriever = IndexBackedRetriever(embedder, index, make_passages(), retrieval_buffer=0)

    passages = retriever.retrieve("query text", top_k=2)

    assert passages[0].passage_id == "p2"
    assert passages[0].source == "BookB"
    assert passages[0].text == "second passage"
    assert passages[0].score == 0.95
    assert passages[1].passage_id == "p3"
    assert passages[1].score == 0.42


def test_retrieve_respects_top_k_smaller_than_available_hits():
    embedder = FakeEmbeddingClient()
    index = FakeVectorIndex([("p1", 0.9), ("p2", 0.8), ("p3", 0.7)])
    retriever = IndexBackedRetriever(embedder, index, make_passages(), retrieval_buffer=0)

    passages = retriever.retrieve("query text", top_k=1)

    assert len(passages) == 1
    assert index.searched_top_k == [1]  # fetch_k = top_k + buffer = 1+0 = 1


def test_retrieve_of_a_query_string_is_a_pure_function_of_its_arguments():
    embedder = FakeEmbeddingClient()
    index = FakeVectorIndex([("p1", 0.9)])
    retriever = IndexBackedRetriever(embedder, index, make_passages(), retrieval_buffer=0)

    first = retriever.retrieve("query text", top_k=1)
    second = retriever.retrieve("query text", top_k=1)

    assert first == second


# ──────────────────────────────────────────
# Tests for Parent-Child RAG retrieval
# ──────────────────────────────────────────

def make_hierarchical_passages():
    """One parent chunk with two children."""
    return {
        # Parent record: no parent_id
        "Book::parent-0": PassageRecord(
            passage_id="Book::parent-0",
            source="BookA",
            text="full parent context with more words than either child",
            parent_id=None,
        ),
        # Child records: point to parent
        "Book::child-0": PassageRecord(
            passage_id="Book::child-0",
            source="BookA",
            text="full parent context",
            parent_id="Book::parent-0",
        ),
        "Book::child-1": PassageRecord(
            passage_id="Book::child-1",
            source="BookA",
            text="with more words than either child",
            parent_id="Book::parent-0",
        ),
    }


def test_parent_child_retrieval_returns_parent_text():
    """When a child chunk is retrieved, the retriever should return the parent's text."""
    embedder = FakeEmbeddingClient()
    index = FakeVectorIndex([("Book::child-0", 0.9)])
    retriever = IndexBackedRetriever(
        embedder, index, make_hierarchical_passages(), retrieval_buffer=0
    )

    passages = retriever.retrieve("query", top_k=1)

    assert len(passages) == 1
    # passage_id is the child's id (for cache key / trace)
    assert passages[0].passage_id == "Book::child-0"
    # but text/source comes from the parent
    assert passages[0].text == "full parent context with more words than either child"
    assert passages[0].source == "BookA"


def test_parent_child_deduplication_collapses_sibling_children():
    """Two children of the same parent should collapse to one result."""
    embedder = FakeEmbeddingClient()
    # FAISS returns both children (both high score)
    index = FakeVectorIndex([("Book::child-0", 0.9), ("Book::child-1", 0.85)])
    retriever = IndexBackedRetriever(
        embedder, index, make_hierarchical_passages(), retrieval_buffer=0
    )

    # top_k=2 but both children resolve to the same parent -> only 1 result
    passages = retriever.retrieve("query", top_k=2)

    assert len(passages) == 1
    assert passages[0].text == "full parent context with more words than either child"


def test_retrieval_buffer_fetches_extra_candidates():
    """With retrieval_buffer=2, fetch_k = top_k + 2."""
    embedder = FakeEmbeddingClient()
    index = FakeVectorIndex([("p1", 0.9)])
    retriever = IndexBackedRetriever(embedder, index, make_passages(), retrieval_buffer=2)

    retriever.retrieve("query", top_k=3)

    # Should have fetched top_k + buffer = 3 + 2 = 5
    assert index.searched_top_k == [5]
