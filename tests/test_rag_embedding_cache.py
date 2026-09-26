from medqa_multiagent.rag.embedding_cache import OnDiskEmbeddingCache

from tests.fakes import FakeEmbeddingClient



def test_embed_query_cache_hit_avoids_a_second_call(tmp_path):
    fake = FakeEmbeddingClient({"hello": [1.0, 2.0]})
    cache = OnDiskEmbeddingCache(fake, tmp_path)

    first = cache.embed_query("hello")
    second = cache.embed_query("hello")

    assert first == [1.0, 2.0]
    assert second == [1.0, 2.0]
    assert fake.embedded_queries == ["hello"]  # only called once


def test_embed_query_cache_miss_for_different_text(tmp_path):
    fake = FakeEmbeddingClient({"hello": [1.0], "goodbye": [2.0]})
    cache = OnDiskEmbeddingCache(fake, tmp_path)

    cache.embed_query("hello")
    cache.embed_query("goodbye")

    assert fake.embedded_queries == ["hello", "goodbye"]


def test_embed_passages_only_computes_uncached_texts(tmp_path):
    fake = FakeEmbeddingClient({"a": [1.0], "b": [2.0], "c": [3.0]})
    cache = OnDiskEmbeddingCache(fake, tmp_path)

    cache.embed_passages(["a", "b"])  # warms the cache for a, b
    result = cache.embed_passages(["a", "b", "c"])

    assert result == [[1.0], [2.0], [3.0]]
    # First call embedded [a, b]; second call should only embed [c].
    assert fake.embedded_passage_batches == [["a", "b"], ["c"]]


def test_embed_passages_preserves_original_order(tmp_path):
    fake = FakeEmbeddingClient({"a": [1.0], "b": [2.0], "c": [3.0]})
    cache = OnDiskEmbeddingCache(fake, tmp_path)

    cache.embed_passages(["b"])  # warm only "b"
    result = cache.embed_passages(["a", "b", "c"])

    assert result == [[1.0], [2.0], [3.0]]


def test_query_and_passage_caches_are_independent(tmp_path):
    fake = FakeEmbeddingClient({"same-text": [9.0]})
    cache = OnDiskEmbeddingCache(fake, tmp_path)

    cache.embed_query("same-text")
    cache.embed_passages(["same-text"])

    # Both hit the underlying client once each -- a query cache entry must
    # not satisfy a passage request for the same text, since MedCPT's
    # query/article encoders are different models.
    assert fake.embedded_queries == ["same-text"]
    assert fake.embedded_passage_batches == [["same-text"]]


def test_cache_persists_across_instances(tmp_path):
    fake_first = FakeEmbeddingClient({"hello": [1.0]})
    OnDiskEmbeddingCache(fake_first, tmp_path).embed_query("hello")

    fake_second = FakeEmbeddingClient({})  # no scripted vector -- would differ if called
    result = OnDiskEmbeddingCache(fake_second, tmp_path).embed_query("hello")

    assert result == [1.0]
    assert fake_second.embedded_queries == []
