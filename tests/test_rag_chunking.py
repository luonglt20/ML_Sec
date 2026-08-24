import pytest

from medqa_multiagent.rag.chunking import (
    approximate_token_count,
    chunk_documents,
    chunk_text,
    chunk_text_hierarchical,
    hierarchical_documents,
)


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


# ──────────────────────────────────────────
# Tests for chunk_text_hierarchical
# ──────────────────────────────────────────


def test_hierarchical_produces_parent_then_children():
    """Parents should precede their children in the returned list."""
    text = " ".join(f"w{i}" for i in range(8))  # 8 words
    # parent_size=8, child_size=4: 1 parent, 2 children
    chunks = chunk_text_hierarchical(text, "Book", child_size=4, parent_size=8)

    parents = [c for c in chunks if c.parent_id is None]
    children = [c for c in chunks if c.parent_id is not None]

    assert len(parents) == 1
    assert len(children) == 2
    # Parents appear before their children
    assert chunks.index(parents[0]) < chunks.index(children[0])


def test_hierarchical_child_parent_id_points_to_parent():
    text = " ".join(f"w{i}" for i in range(8))
    chunks = chunk_text_hierarchical(text, "Book", child_size=4, parent_size=8)

    parent = next(c for c in chunks if c.parent_id is None)
    children = [c for c in chunks if c.parent_id is not None]

    assert all(child.parent_id == parent.chunk_id for child in children)


def test_hierarchical_parent_contains_full_window():
    words = [f"w{i}" for i in range(8)]
    text = " ".join(words)
    chunks = chunk_text_hierarchical(text, "Book", child_size=4, parent_size=8)

    parent = next(c for c in chunks if c.parent_id is None)
    # Parent should contain all 8 words
    assert parent.text == text


def test_hierarchical_children_cover_parent_words():
    words = [f"w{i}" for i in range(8)]
    text = " ".join(words)
    chunks = chunk_text_hierarchical(text, "Book", child_size=4, parent_size=8)

    children = [c for c in chunks if c.parent_id is not None]
    combined = " ".join(c.text for c in children)
    assert combined == text


def test_hierarchical_assigns_stable_ids():
    text = " ".join(f"w{i}" for i in range(8))
    chunks = chunk_text_hierarchical(text, "Anatomy", child_size=4, parent_size=8)

    parent = next(c for c in chunks if c.parent_id is None)
    children = [c for c in chunks if c.parent_id is not None]

    assert parent.chunk_id == "Anatomy::parent-0"
    assert [c.chunk_id for c in children] == ["Anatomy::child-0", "Anatomy::child-1"]


def test_hierarchical_empty_text_yields_no_chunks():
    assert chunk_text_hierarchical("", "Book", child_size=4, parent_size=8) == []
    assert chunk_text_hierarchical("   ", "Book", child_size=4, parent_size=8) == []


def test_hierarchical_raises_on_invalid_sizes():
    with pytest.raises(ValueError):
        chunk_text_hierarchical("text", "Book", child_size=0, parent_size=8)
    with pytest.raises(ValueError):
        chunk_text_hierarchical("text", "Book", child_size=4, parent_size=0)
    with pytest.raises(ValueError):  # parent < child
        chunk_text_hierarchical("text", "Book", child_size=8, parent_size=4)


def test_hierarchical_is_deterministic():
    text = " ".join(f"w{i}" for i in range(20))
    first = chunk_text_hierarchical(text, "Book", child_size=4, parent_size=8)
    second = chunk_text_hierarchical(text, "Book", child_size=4, parent_size=8)
    assert first == second


def test_hierarchical_multi_parent():
    """Text spanning > 1 parent window produces multiple parents & children."""
    text = " ".join(f"w{i}" for i in range(16))  # 16 words
    # parent_size=8 → 2 parents; child_size=4 → 2 children per parent
    chunks = chunk_text_hierarchical(text, "Book", child_size=4, parent_size=8)

    parents = [c for c in chunks if c.parent_id is None]
    children = [c for c in chunks if c.parent_id is not None]

    assert len(parents) == 2
    assert len(children) == 4

    # Each parent owns exactly 2 children
    children_of_p0 = [c for c in children if c.parent_id == "Book::parent-0"]
    children_of_p1 = [c for c in children if c.parent_id == "Book::parent-1"]
    assert len(children_of_p0) == 2
    assert len(children_of_p1) == 2


def test_hierarchical_documents_wraps_chunk_text_hierarchical():
    documents = [("A", "a0 a1 a2 a3"), ("B", "b0 b1 b2 b3")]
    chunks = hierarchical_documents(documents, child_size=2, parent_size=4)

    parent_ids = {c.chunk_id for c in chunks if c.parent_id is None}
    assert "A::parent-0" in parent_ids
    assert "B::parent-0" in parent_ids
    # All children have a parent_id pointing to a valid parent
    for c in chunks:
        if c.parent_id is not None:
            assert c.parent_id in parent_ids
