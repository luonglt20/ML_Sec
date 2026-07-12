"""Builds the one standard RAG retriever stack shared by every caller.

Mirrors `medqa_multiagent.client_factory.build_llm_client`: every caller
that needs a *real* (non-fake) `Retriever` -- `entrypoint.answer_question`
(when no retriever is explicitly injected), the CLI, and the demo UI --
gets it from exactly this one function, so they never silently drift
apart and so a query already answered via one caller is a retrieval-cache
hit when re-submitted through another.

Loads the pre-built FAISS index + passage metadata written once by
`scripts/build_rag_index.py`, wraps the MedCPT embedding client in an
on-disk cache, and wraps the resulting retriever in an on-disk retrieval
cache -- so a fully-assembled `Retriever` is ready to call `.retrieve()`
on immediately.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Union

from ..config import RunConfig
from .corpus import read_chunks
from .embedding_cache import OnDiskEmbeddingCache
from .embeddings import EmbeddingClient, MedCptEmbeddingClient
from .index import FaissFlatIndex, PassageRecord
from .retrieval_cache import OnDiskRetrievalCache
from .retriever import IndexBackedRetriever, Retriever

#: Default on-disk cache directories, shared by every caller unless overridden.
DEFAULT_EMBEDDING_CACHE_DIR = ".cache/embeddings"
DEFAULT_RETRIEVAL_CACHE_DIR = ".cache/retrieval"

#: Filename (within an index directory) holding passage metadata
#: (id/source/text), written by `scripts/build_rag_index.py` via
#: `corpus.write_chunks`.
PASSAGES_FILENAME = "passages.jsonl"


def _load_passages(index_dir: Path) -> dict:
    passages_path = index_dir / PASSAGES_FILENAME
    if not passages_path.exists():
        raise FileNotFoundError(
            f"No RAG passage metadata found at {passages_path}. "
            "Run scripts/build_rag_index.py first."
        )
    chunks = read_chunks(passages_path)
    return {
        chunk.chunk_id: PassageRecord(
            passage_id=chunk.chunk_id, source=chunk.source, text=chunk.text
        )
        for chunk in chunks
    }


@functools.lru_cache(maxsize=None)
def _build_retriever_cached(
    index_dir: str, embedding_cache_dir: str, retrieval_cache_dir: str
) -> Retriever:
    """Build (and process-wide cache) the retriever stack for one index dir.

    Loading the FAISS index and MedCPT model weights is expensive, so this
    is memoized per `(index_dir, embedding_cache_dir, retrieval_cache_dir)`
    -- repeated calls (e.g. once per question in a dev-set run) reuse the
    same in-memory retriever rather than reloading the index/model each time.
    """
    index_path = Path(index_dir)
    passages = _load_passages(index_path)
    vector_index = FaissFlatIndex.load(index_path)
    embedding_client: EmbeddingClient = OnDiskEmbeddingCache(
        MedCptEmbeddingClient(), embedding_cache_dir
    )
    retriever = IndexBackedRetriever(embedding_client, vector_index, passages)
    return OnDiskRetrievalCache(retriever, retrieval_cache_dir, index_id=str(index_path))


def build_retriever(
    config: RunConfig,
    index_dir: Union[str, Path, None] = None,
    embedding_cache_dir: Union[str, Path] = DEFAULT_EMBEDDING_CACHE_DIR,
    retrieval_cache_dir: Union[str, Path] = DEFAULT_RETRIEVAL_CACHE_DIR,
) -> Retriever:
    """Build the standard cache-wrapped `Retriever` for `config.rag_index_dir`.

    Args:
        config: This run's configuration; `config.rag_index_dir` is used
            unless `index_dir` overrides it.
        index_dir: Directory holding the pre-built FAISS index + passage
            metadata (default: `config.rag_index_dir`).
        embedding_cache_dir: On-disk cache directory for embedding calls.
        retrieval_cache_dir: On-disk cache directory for retrieval calls.

    Raises:
        FileNotFoundError: if no pre-built index/passage metadata exists
            at the resolved `index_dir` -- run `scripts/build_rag_index.py`
            first.
        ImportError: if `faiss`/`transformers`/`torch` aren't installed
            (install the `rag` extra: `pip install -e ".[rag]"`).
    """
    resolved_index_dir = str(index_dir if index_dir is not None else config.rag_index_dir)
    return _build_retriever_cached(
        resolved_index_dir, str(embedding_cache_dir), str(retrieval_cache_dir)
    )
