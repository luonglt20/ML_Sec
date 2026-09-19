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
import os
import platform
import time
from pathlib import Path
from typing import Optional, Union

from ..config import RunConfig
from .corpus import read_chunks
from .embedding_cache import OnDiskEmbeddingCache
from .embeddings import EmbeddingClient, MedCptEmbeddingClient
from .index import FaissFlatIndex, NumpyFlatIndex, PassageRecord
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
            passage_id=chunk.chunk_id,
            source=chunk.source,
            text=chunk.text,
            parent_id=chunk.parent_id,  # None for flat chunks; set for child chunks
        )
        for chunk in chunks
    }


@functools.lru_cache(maxsize=None)
def _build_retriever_cached(
    index_dir: str,
    embedding_cache_dir: str,
    retrieval_cache_dir: str,
    retrieval_buffer: int,
    use_hybrid: bool,
    relevance_threshold: float,
    use_option_boosting: bool,
    option_boost_weight: float,
    dynamic_top_k: bool,
    heuristic_compression: bool,
    use_reranker: bool,
    use_query_pruning: bool,
    use_synonym_expansion: bool,
    use_mmr: bool = False,
    mmr_lambda: float = 0.7,
    use_hyde: bool = False,
    vector_backend: str = "auto",
) -> Retriever:
    """Build (and process-wide cache) the retriever stack for one index dir."""
    _t0 = time.monotonic()
    index_path = Path(index_dir)
    passages = _load_passages(index_path)
    resolved_backend = vector_backend
    if resolved_backend == "auto":
        # The macOS faiss-cpu and PyTorch wheels can bundle incompatible
        # libomp copies. Prefer the persisted NumPy matrix there to prevent a
        # native abort/segfault at the first search.
        has_numpy_index = (index_path / "vectors.npy").exists()
        resolved_backend = (
            "numpy" if platform.system() == "Darwin" and has_numpy_index else "faiss"
        )
    if resolved_backend == "numpy":
        vector_index = NumpyFlatIndex.load(index_path)
        print("[RAG] Using NumPy exact-search backend (macOS-safe)", flush=True)
    elif resolved_backend == "faiss":
        vector_index = FaissFlatIndex.load(index_path)
    else:
        raise ValueError("MEDQA_VECTOR_BACKEND must be 'auto', 'numpy', or 'faiss'")

    bm25_index = None
    if use_hybrid:
        bm25_path = index_path / "bm25.json"
        if bm25_path.exists():
            from medqa_multiagent.rag.bm25 import BM25Index
            bm25_index = BM25Index.load(bm25_path)
            print(f"[RAG] Loaded BM25 sparse index from {bm25_path}", flush=True)
        else:
            print(f"[RAG] Warning: use_hybrid is True but BM25 index not found at {bm25_path}", flush=True)

    embedding_client: EmbeddingClient = OnDiskEmbeddingCache(
        MedCptEmbeddingClient(), embedding_cache_dir
    )
    retriever = IndexBackedRetriever(
        embedding_client,
        vector_index,
        passages,
        retrieval_buffer=retrieval_buffer,
        bm25_index=bm25_index,
        relevance_threshold=relevance_threshold,
        use_option_boosting=use_option_boosting,
        option_boost_weight=option_boost_weight,
        dynamic_top_k=dynamic_top_k,
        heuristic_compression=heuristic_compression,
        use_reranker=use_reranker,
        use_query_pruning=use_query_pruning,
        use_synonym_expansion=use_synonym_expansion,
        use_mmr=use_mmr,
        mmr_lambda=mmr_lambda,
        use_hyde=use_hyde,
    )
    _elapsed = time.monotonic() - _t0
    mode_str = "hybrid" if bm25_index is not None else "dense"
    print(
        f"[RAG] Retriever loaded in {_elapsed:.1f}s "
        f"({len(passages)} passages, mode={mode_str}, threshold={relevance_threshold})",
        flush=True,
    )
    return OnDiskRetrievalCache(retriever, retrieval_cache_dir, index_id=str(index_path))


def build_retriever(
    config: RunConfig,
    index_dir: Union[str, Path, None] = None,
    embedding_cache_dir: Union[str, Path] = DEFAULT_EMBEDDING_CACHE_DIR,
    retrieval_cache_dir: Union[str, Path] = DEFAULT_RETRIEVAL_CACHE_DIR,
    vector_backend: Optional[str] = None,
) -> Retriever:
    """Build the standard cache-wrapped `Retriever` for `config.rag_index_dir`."""
    resolved_index_dir = str(index_dir if index_dir is not None else config.rag_index_dir)
    resolved_backend = vector_backend or os.environ.get("MEDQA_VECTOR_BACKEND", "auto")
    return _build_retriever_cached(
        resolved_index_dir,
        str(embedding_cache_dir),
        str(retrieval_cache_dir),
        config.rag_retrieval_buffer,
        config.rag_use_hybrid,
        config.rag_relevance_threshold,
        config.rag_use_option_boosting,
        config.rag_option_boost_weight,
        config.rag_dynamic_top_k,
        config.rag_heuristic_compression,
        config.rag_use_reranker,
        config.rag_use_query_pruning,
        config.rag_use_synonym_expansion,
        getattr(config, "rag_use_mmr", False),
        getattr(config, "rag_mmr_lambda", 0.7),
        getattr(config, "rag_use_hyde", False),
        resolved_backend.lower(),
    )
