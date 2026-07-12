import pytest

from medqa_multiagent.rag.chunking import approximate_token_count, chunk_documents, chunk_text


def test_approximate_token_count_counts_whitespace_delimited_words():
    assert approximate_token_count("one two three") == 3
    assert approximate_token_count("") == 0
    assert approximate_token_count("   ") == 0


def test_chunk_text_splits_into_fixed_size_pieces():
    text = " ".join(f"word{i}" for i in range(10))
    chunks = chunk_text(text, source="book", chunk_size_tokens=4)

    assert [chunk.text for chunk in chunks] == [
        "word0 word1 word2 word3",
        "word4 word5 word6 word7",
        "word8 word9",
    ]


def test_chunk_text_assigns_stable_source_derived_ids():
    text = " ".join(f"word{i}" for i in range(8))
    chunks = chunk_text(text, source="Anatomy_Gray", chunk_size_tokens=4)

    assert [chunk.chunk_id for chunk in chunks] == [
        "Anatomy_Gray::chunk-0",
        "Anatomy_Gray::chunk-1",
    ]
    assert all(chunk.source == "Anatomy_Gray" for chunk in chunks)


def test_chunk_text_of_empty_text_yields_no_chunks():
    assert chunk_text("", source="book", chunk_size_tokens=256) == []
    assert chunk_text("   ", source="book", chunk_size_tokens=256) == []


def test_chunk_text_rejects_non_positive_chunk_size():
    with pytest.raises(ValueError):
        chunk_text("word", source="book", chunk_size_tokens=0)
    with pytest.raises(ValueError):
        chunk_text("word", source="book", chunk_size_tokens=-1)


def test_chunk_text_is_deterministic():
    text = " ".join(f"word{i}" for i in range(20))
    first = chunk_text(text, source="book", chunk_size_tokens=7)
    second = chunk_text(text, source="book", chunk_size_tokens=7)
    assert first == second


def test_chunk_documents_preserves_document_order_and_chunks_each():
    documents = [
        ("BookA", "a0 a1 a2 a3"),
        ("BookB", "b0 b1"),
    ]
    chunks = chunk_documents(documents, chunk_size_tokens=2)

    assert [c.chunk_id for c in chunks] == [
        "BookA::chunk-0",
        "BookA::chunk-1",
        "BookB::chunk-0",
    ]
