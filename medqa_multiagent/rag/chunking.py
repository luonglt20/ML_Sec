"""Deterministic, dependency-free corpus chunking.

Splits a document's raw text into fixed-size (~`chunk_size_tokens`-token)
non-overlapping chunks. Token count is approximated by whitespace-splitting
(a `tiktoken`/model-specific tokenizer would be more precise, but this
project's chunk-size target is an approximate, reproducibility-oriented
knob -- see `DESIGN.md` SS6 -- not a billing-accurate token count, so a
zero-dependency approximation is deliberately used here instead of adding a
tokenizer dependency).

`chunk_text_hierarchical` extends the flat chunking with a two-level
Parent-Child scheme suited for RAG retrieval:
  - *Child* chunks (small, ~64 tokens) are embedded and indexed in FAISS --
    smaller windows give the embedding model a tighter, less noisy context
    for higher retrieval precision.
  - *Parent* chunks (large, ~512 tokens) are stored in passage metadata and
    returned to the LLM when a child is retrieved -- larger windows give the
    LLM fuller clinical context without needing to retrieve adjacent chunks
    separately.

Pure logic, no network/embedding/index calls -- fully unit-testable on its
own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Chunk:
    """One fixed-size chunk of a source document.

    Attributes:
        chunk_id: Stable identifier, derived from `source` and the chunk's
            position within it (e.g. ``"Anatomy_Gray::chunk-0"``) -- stable
            across repeated indexing runs given the same input documents
            and chunk size.
        source: The document (e.g. textbook name) this chunk was cut from.
        text: The chunk's text content.
        parent_id: For Parent-Child RAG child chunks, the ``chunk_id`` of the
            enclosing parent chunk (e.g. ``"Anatomy_Gray::parent-0"``). When
            the retriever fetches a child chunk, it resolves this id to the
            parent's larger text to return fuller context to the LLM. ``None``
            for flat chunks produced by ``chunk_text`` / ``chunk_documents``,
            and for parent chunks produced by ``chunk_text_hierarchical``.
    """

    chunk_id: str
    source: str
    text: str
    parent_id: Optional[str] = None


def approximate_token_count(text: str) -> int:
    """Approximate the number of tokens in `text`.

    Whitespace-based word count -- a deliberately simple, zero-dependency
    proxy for token count (see module docstring). Used only to decide
    where one ~`chunk_size_tokens` chunk ends and the next begins, not for
    any cost/billing calculation (that uses the LLM provider's own
    reported `prompt_tokens`/`completion_tokens`, see `llm_client.py`).
    """
    return len(text.split())


def chunk_text(text: str, source: str, chunk_size_tokens: int) -> List[Chunk]:
    """Split `text` into fixed-size, non-overlapping chunks of `source`.

    Chunks are formed by greedily grouping whitespace-delimited words until
    `chunk_size_tokens` is reached (the last chunk may be shorter). A pure
    function of its arguments: identical inputs always yield identical
    chunks in the same order, with the same `chunk_id`s.

    Raises:
        ValueError: if `chunk_size_tokens` is not a positive integer.
    """
    if chunk_size_tokens <= 0:
        raise ValueError("chunk_size_tokens must be a positive integer")

    words = text.split()
    if not words:
        return []

    chunks: List[Chunk] = []
    for index, start in enumerate(range(0, len(words), chunk_size_tokens)):
        piece_words = words[start : start + chunk_size_tokens]
        chunks.append(
            Chunk(
                chunk_id=f"{source}::chunk-{index}",
                source=source,
                text=" ".join(piece_words),
            )
        )
    return chunks


def chunk_documents(
    documents: Sequence[Tuple[str, str]], chunk_size_tokens: int
) -> List[Chunk]:
    """Chunk every `(source, text)` document, preserving document order.

    Args:
        documents: Sequence of `(source, text)` pairs, e.g. one per
            textbook, in the order they should be chunked/indexed.
        chunk_size_tokens: Target chunk size in (approximate) tokens,
            applied uniformly to every document.
    """
    all_chunks: List[Chunk] = []
    for source, text in documents:
        all_chunks.extend(chunk_text(text, source, chunk_size_tokens))
    return all_chunks


def chunk_text_hierarchical(
    text: str,
    source: str,
    child_size: int,
    parent_size: int,
) -> List[Chunk]:
    """Split `text` into a two-level Parent-Child chunk hierarchy.

    Produces a flat list of all chunks -- parents first, then their
    children. Only *child* chunks (``parent_id is not None``) should be
    embedded and indexed in FAISS. When a child hit is retrieved at query
    time, the retriever resolves ``child.parent_id`` to the corresponding
    parent chunk and returns the parent's (larger) text as context.

    Layout::

        words[0..parent_size-1]         -> parent-0 (returned to LLM)
          words[0..child_size-1]        -> child-0  (embedded in FAISS)
          words[child_size..2*child-1]  -> child-1
          ...
        words[parent_size..2*parent-1]  -> parent-1
          ...

    Args:
        text: Raw document text to chunk.
        source: Document identifier (e.g. textbook title).
        child_size: Size of child chunks in approximate tokens.
        parent_size: Size of parent chunks in approximate tokens.
            Should be a whole multiple of ``child_size`` for clean coverage,
            but any positive integer >= ``child_size`` is accepted.

    Raises:
        ValueError: if ``child_size`` or ``parent_size`` are not positive,
            or if ``parent_size < child_size``.
    """
    if child_size <= 0:
        raise ValueError("child_size must be a positive integer")
    if parent_size <= 0:
        raise ValueError("parent_size must be a positive integer")
    if parent_size < child_size:
        raise ValueError("parent_size must be >= child_size")

    words = text.split()
    if not words:
        return []

    all_chunks: List[Chunk] = []
    parent_index = 0
    child_index = 0

    for p_start in range(0, len(words), parent_size):
        parent_words = words[p_start : p_start + parent_size]
        parent_id = f"{source}::parent-{parent_index}"

        # Parent chunk -- NOT indexed in FAISS, only stored for context lookup.
        all_chunks.append(
            Chunk(
                chunk_id=parent_id,
                source=source,
                text=" ".join(parent_words),
                parent_id=None,  # parents have no parent of their own
            )
        )

        # Child chunks within this parent's word range -- these are indexed.
        for c_start in range(0, len(parent_words), child_size):
            child_words = parent_words[c_start : c_start + child_size]
            all_chunks.append(
                Chunk(
                    chunk_id=f"{source}::child-{child_index}",
                    source=source,
                    text=" ".join(child_words),
                    parent_id=parent_id,
                )
            )
            child_index += 1

        parent_index += 1

    return all_chunks


def hierarchical_documents(
    documents: Sequence[Tuple[str, str]],
    child_size: int,
    parent_size: int,
) -> List[Chunk]:
    """Apply ``chunk_text_hierarchical`` to every ``(source, text)`` document.

    Convenience wrapper that mirrors ``chunk_documents`` for the hierarchical
    case -- used by ``scripts/build_rag_index.py``.
    """
    all_chunks: List[Chunk] = []
    for source, text in documents:
        all_chunks.extend(chunk_text_hierarchical(text, source, child_size, parent_size))
    return all_chunks
