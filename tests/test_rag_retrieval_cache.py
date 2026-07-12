from medqa_multiagent.rag.retrieval_cache import OnDiskRetrievalCache
from medqa_multiagent.rag.retriever import Passage

from fakes import FakeRetriever


def make_passages():
    return [
        Passage(passage_id="p1", source="BookA", text="first passage", score=0.9),
        Passage(passage_id="p2", source="BookB", text="second passage", score=0.5),
    ]


def test_resubmitting_the_same_query_is_a_cache_hit(tmp_path):
    fake = FakeRetriever(make_passages())
    cache = OnDiskRetrievalCache(fake, tmp_path, index_id="data/rag_index")

    first = cache.retrieve("what causes X?", top_k=2)
    second = cache.retrieve("what causes X?", top_k=2)

    assert first == second == make_passages()
    assert fake.calls == [("what causes X?", 2)]  # only called once


def test_differing_query_is_a_cache_miss(tmp_path):
    fake = FakeRetriever(make_passages())
    cache = OnDiskRetrievalCache(fake, tmp_path, index_id="data/rag_index")

    cache.retrieve("query a", top_k=2)
    cache.retrieve("query b", top_k=2)

    assert len(fake.calls) == 2


def test_differing_top_k_is_a_cache_miss(tmp_path):
    fake = FakeRetriever(make_passages())
    cache = OnDiskRetrievalCache(fake, tmp_path, index_id="data/rag_index")

    cache.retrieve("query", top_k=1)
    cache.retrieve("query", top_k=2)

    assert len(fake.calls) == 2


def test_differing_index_id_is_a_cache_miss(tmp_path):
    fake = FakeRetriever(make_passages())
    OnDiskRetrievalCache(fake, tmp_path, index_id="index-v1").retrieve("query", top_k=2)
    OnDiskRetrievalCache(fake, tmp_path, index_id="index-v2").retrieve("query", top_k=2)

    assert len(fake.calls) == 2


def test_cache_persists_across_instances(tmp_path):
    first_retriever = FakeRetriever(make_passages())
    OnDiskRetrievalCache(first_retriever, tmp_path, index_id="idx").retrieve("query", top_k=2)

    second_retriever = FakeRetriever([])  # would return nothing if actually called
    result = OnDiskRetrievalCache(second_retriever, tmp_path, index_id="idx").retrieve(
        "query", top_k=2
    )

    assert result == make_passages()
    assert second_retriever.calls == []
