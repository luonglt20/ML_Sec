"""Deterministic, dependency-free corpus chunking.

Splits a document's raw text into fixed-size (~`chunk_size_tokens`-token)
non-overlapping chunks. Token count is approximated by whitespace-splitting
(a `tiktoken`/model-specific tokenizer would be more precise, but this
project's chunk-size target is an approximate, reproducibility-oriented
knob -- see `DESIGN.md` SS6 -- not a billing-accurate token count, so a
zero-dependency approximation is deliberately used here instead of adding a
tokenizer dependency).

Pure logic, no network/embedding/index calls -- fully unit-testable on its
own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple


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
    """

    chunk_id: str
    source: str
    text: str


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
