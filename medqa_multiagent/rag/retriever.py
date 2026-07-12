"""The query-time retrieval interface: `Retriever` and `IndexBackedRetriever`.

`Retriever` is the RAG module's outward-facing dependency-injection seam --
mirroring `llm_client.LLMClient` -- that `entrypoint.answer_question`
programs against for V1 (and every later variant that retains RAG).
Tests substitute a scripted `FakeRetriever` (see `tests/fakes.py`);
`IndexBackedRetriever` is the real implementation, composed from an
`embeddings.EmbeddingClient` (to embed the query) and an `index.VectorIndex`
(to search it) plus a passage-metadata lookup -- both of which are
*themselves* fakeable, so `IndexBackedRetriever` itself is fully
unit-testable without FAISS or MedCPT installed (see
`tests/rag/test_retriever.py`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, Protocol

from .embeddings import EmbeddingClient
from .index import PassageRecord, VectorIndex


@dataclass(frozen=True)
class Passage:
    """One retrieved passage, as recorded in a prediction's trace.

    Attributes:
        passage_id: Stable identifier of the retrieved passage/chunk.
        source: The document (e.g. textbook name) this passage was cut from.
        text: The passage's text content.
        score: The retriever's similarity score for this passage against
            the query that retrieved it (higher is more similar, for the
            inner-product-based `FaissFlatIndex`).
    """

    passage_id: str
    source: str
    text: str
    score: float


class Retriever(Protocol):
    """The one interface every RAG-using variant retrieves passages through."""

    def retrieve(self, query: str, top_k: int) -> List[Passage]: ...


class IndexBackedRetriever:
    """`Retriever` composed from an `EmbeddingClient` and a `VectorIndex`.

    Embeds `query` via `embedding_client.embed_query`, searches
    `vector_index` for the `top_k` nearest passage vectors, and resolves
    each hit's `passage_id` to its text/source via `passages`. Passages are
    returned in the index's own ranked (best-first) order.

    Raises:
        KeyError: if the vector index returns a `passage_id` not present
            in `passages` -- indicates the index and passage-metadata
            file are out of sync (e.g. rebuilt independently of one
            another), a corpus-integrity bug that should surface loudly
            rather than silently drop the mismatched result.
    """

    def __init__(
        self,
        embedding_client: EmbeddingClient,
        vector_index: VectorIndex,
        passages: Mapping[str, PassageRecord],
    ) -> None:
        self._embedding_client = embedding_client
        self._vector_index = vector_index
        self._passages = passages

    def retrieve(self, query: str, top_k: int) -> List[Passage]:
        query_vector = self._embedding_client.embed_query(query)
        hits = self._vector_index.search(query_vector, top_k)

        results: List[Passage] = []
        for passage_id, score in hits:
            record = self._passages[passage_id]
            results.append(
                Passage(
                    passage_id=record.passage_id,
                    source=record.source,
                    text=record.text,
                    score=score,
                )
            )
        return results
