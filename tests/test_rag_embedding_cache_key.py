from medqa_multiagent.rag.embedding_cache_key import compute_embedding_cache_key


def test_identical_requests_produce_the_same_key():
    a = compute_embedding_cache_key("query", "hello", "model-x")
    b = compute_embedding_cache_key("query", "hello", "model-x")
    assert a == b


def test_differing_kind_changes_the_key():
    query_key = compute_embedding_cache_key("query", "hello", "model-x")
    passage_key = compute_embedding_cache_key("passage", "hello", "model-x")
    assert query_key != passage_key


def test_differing_text_changes_the_key():
    a = compute_embedding_cache_key("query", "hello", "model-x")
    b = compute_embedding_cache_key("query", "goodbye", "model-x")
    assert a != b


def test_differing_model_name_changes_the_key():
    a = compute_embedding_cache_key("query", "hello", "model-x")
    b = compute_embedding_cache_key("query", "hello", "model-y")
    assert a != b


def test_key_is_a_64_char_hex_digest():
    key = compute_embedding_cache_key("query", "hello", "model-x")
    assert len(key) == 64
    int(key, 16)  # raises ValueError if not valid hex
