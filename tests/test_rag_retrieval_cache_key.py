from medqa_multiagent.rag.retrieval_cache_key import compute_retrieval_cache_key


def test_identical_requests_produce_the_same_key():
    a = compute_retrieval_cache_key("what causes X?", 3, "data/rag_index")
    b = compute_retrieval_cache_key("what causes X?", 3, "data/rag_index")
    assert a == b


def test_differing_query_changes_the_key():
    a = compute_retrieval_cache_key("query a", 3, "data/rag_index")
    b = compute_retrieval_cache_key("query b", 3, "data/rag_index")
    assert a != b


def test_differing_top_k_changes_the_key():
    a = compute_retrieval_cache_key("query", 3, "data/rag_index")
    b = compute_retrieval_cache_key("query", 5, "data/rag_index")
    assert a != b


def test_differing_index_id_changes_the_key():
    a = compute_retrieval_cache_key("query", 3, "data/rag_index_v1")
    b = compute_retrieval_cache_key("query", 3, "data/rag_index_v2")
    assert a != b


def test_differing_options_change_key_but_mapping_order_does_not():
    a = compute_retrieval_cache_key(
        "query", 3, "index", {"A": "alpha", "B": "beta"}
    )
    reordered = compute_retrieval_cache_key(
        "query", 3, "index", {"B": "beta", "A": "alpha"}
    )
    changed = compute_retrieval_cache_key(
        "query", 3, "index", {"A": "alpha", "B": "changed"}
    )
    assert a == reordered
    assert a != changed


def test_key_is_a_64_char_hex_digest():
    key = compute_retrieval_cache_key("query", 3, "data/rag_index")
    assert len(key) == 64
    int(key, 16)
