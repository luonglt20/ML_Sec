"""The vector index interface, and its FAISS flat-index implementation.

`VectorIndex` is a small `Protocol` (nearest-neighbor search over
already-embedded vectors) that `rag.retriever.IndexBackedRetriever`
programs against -- the second half of the RAG module's dependency-
injection seam alongside `embeddings.EmbeddingClient`. Tests substitute a
`FakeVectorIndex` (see `tests/fakes.py`); `FaissFlatIndex` is the real
implementation, backed by a FAISS `IndexFlatIP` (exact inner-product
search -- see DESIGN.md SS6: "no approximate indexing, corpus is small
enough").

`faiss` is imported lazily, inside `FaissFlatIndex`'s methods, so importing
this module never requires that optional, heavy dependency unless a real
(non-fake) index is actually built/loaded/searched. Install it via the
`rag` extra (`pip install -e ".[rag]"`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Protocol, Sequence, Tuple, Union

_INDEX_FILENAME = "index.faiss"
_IDS_FILENAME = "passage_ids.json"


@dataclass(frozen=True)
class PassageRecord:
    """One indexed passage's metadata (text + provenance, no vector).

    Attributes:
        passage_id: Stable identifier (matches `chunking.Chunk.chunk_id`).
        source: The document (e.g. textbook name) this passage was cut from.
        text: The passage's text content.
    """

    passage_id: str
    source: str
    text: str


class VectorIndex(Protocol):
    """The one interface the RAG module searches embedded vectors through."""

    def search(
        self, query_vector: Sequence[float], top_k: int
    ) -> List[Tuple[str, float]]:
        """Return up to `top_k` `(passage_id, score)` pairs, best first."""
        ...


class FaissFlatIndex:
    """`VectorIndex` backed by a FAISS `IndexFlatIP` (exact search).

    Built once (via `build`) from every passage's embedding vector and its
    `passage_id`, then persisted to disk (`save`) so
    `scripts/build_rag_index.py` only ever needs to run once; loaded back
    (`load`) at query time by `rag.client_factory.build_retriever`.
    """

    def __init__(self, index, passage_ids: List[str]) -> None:
        self._index = index
        self._passage_ids = passage_ids

    @classmethod
    def build(
        cls, vectors: Sequence[Sequence[float]], passage_ids: Sequence[str]
    ) -> "FaissFlatIndex":
        """Build a fresh flat index from parallel `vectors`/`passage_ids`.

        Raises:
            ValueError: if `vectors` and `passage_ids` have different
                lengths, or `vectors` is empty.
        """
        import faiss  # lazy import -- see module docstring
        import numpy as np

        if len(vectors) != len(passage_ids):
            raise ValueError(
                f"vectors ({len(vectors)}) and passage_ids ({len(passage_ids)}) "
                "must have the same length"
            )
        if not vectors:
            raise ValueError("Cannot build an index from zero vectors")

        matrix = np.asarray(vectors, dtype="float32")
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)  # type: ignore[call-arg]  # faiss's SWIG-generated stubs mis-describe this call
        return cls(index, list(passage_ids))

    def search(
        self, query_vector: Sequence[float], top_k: int
    ) -> List[Tuple[str, float]]:
        import numpy as np

        query = np.asarray([query_vector], dtype="float32")
        k = min(top_k, len(self._passage_ids))
        scores, indices = self._index.search(query, k)
        results: List[Tuple[str, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS pads with -1 if fewer than k results exist
                continue
            results.append((self._passage_ids[idx], float(score)))
        return results

    def save(self, directory: Union[str, Path]) -> None:
        """Persist this index's FAISS state and passage-id ordering to disk."""
        import faiss  # lazy import -- see module docstring

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(directory / _INDEX_FILENAME))
        (directory / _IDS_FILENAME).write_text(
            json.dumps(self._passage_ids, ensure_ascii=True), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Union[str, Path]) -> "FaissFlatIndex":
        """Load a previously-`save`d index back from disk.

        Raises:
            FileNotFoundError: if `directory` doesn't contain a
                previously-saved index.
        """
        import faiss  # lazy import -- see module docstring

        directory = Path(directory)
        index_path = directory / _INDEX_FILENAME
        ids_path = directory / _IDS_FILENAME
        if not index_path.exists() or not ids_path.exists():
            raise FileNotFoundError(
                f"No RAG index found at {directory}. "
                "Run scripts/build_rag_index.py first."
            )
        index = faiss.read_index(str(index_path))
        passage_ids = json.loads(ids_path.read_text(encoding="utf-8"))
        return cls(index, passage_ids)
