from medqa_multiagent.rag.index import PassageRecord
from medqa_multiagent.rag.retriever import IndexBackedRetriever

from fakes import FakeEmbeddingClient, FakeVectorIndex


def make_passages():
    return {
        "p1": PassageRecord(passage_id="p1", source="BookA", text="first passage"),
        "p2": PassageRecord(passage_id="p2", source="BookB", text="second passage"),
        "p3": PassageRecord(passage_id="p3", source="BookC", text="third passage"),
    }


def test_retrieve_embeds_the_query_and_searches_the_index():
    embedder = FakeEmbeddingClient({"what causes X?": [1.0, 2.0]})
    index = FakeVectorIndex([("p1", 0.9), ("p2", 0.5)])
    retriever = IndexBackedRetriever(embedder, index, make_passages())

    passages = retriever.retrieve("what causes X?", top_k=2)

    assert embedder.embedded_queries == ["what causes X?"]
    assert index.searched_vectors == [[1.0, 2.0]]
    assert index.searched_top_k == [2]
    assert [p.passage_id for p in passages] == ["p1", "p2"]


def test_retrieve_resolves_passage_metadata_and_score_in_ranked_order():
    embedder = FakeEmbeddingClient()
    index = FakeVectorIndex([("p2", 0.95), ("p3", 0.42)])
    retriever = IndexBackedRetriever(embedder, index, make_passages())

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
    retriever = IndexBackedRetriever(embedder, index, make_passages())

    passages = retriever.retrieve("query text", top_k=1)

    assert len(passages) == 1
    assert index.searched_top_k == [1]


def test_retrieve_of_a_query_string_is_a_pure_function_of_its_arguments():
    embedder = FakeEmbeddingClient()
    index = FakeVectorIndex([("p1", 0.9)])
    retriever = IndexBackedRetriever(embedder, index, make_passages())

    first = retriever.retrieve("query text", top_k=1)
    second = retriever.retrieve("query text", top_k=1)

    assert first == second
