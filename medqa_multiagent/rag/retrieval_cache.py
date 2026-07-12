"""On-disk cache wrapping the retrieval interface.

Mirrors `medqa_multiagent.cache.OnDiskLLMCache`: keyed via
`retrieval_cache_key.compute_retrieval_cache_key` (query text, `top_k`,
and an `index_id` identifying the index/corpus version), so re-running an
evaluation after a non-retrieval code change doesn't re-embed/re-search
identical queries, and re-running against a *different* index/corpus
build is a guaranteed miss.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import List, Union

from .retrieval_cache_key import compute_retrieval_cache_key
from .retriever import Passage, Retriever


class OnDiskRetrievalCache:
    """`Retriever` decorator backed by one JSON file per cache key on disk.

    Args:
        retriever: The `Retriever` to wrap.
        cache_dir: Directory to persist cache entries under.
        index_id: Stable identifier for the index/corpus version this
            retriever queries (see `retrieval_cache_key`).
    """

    def __init__(
        self, retriever: Retriever, cache_dir: Union[str, Path], index_id: str
    ) -> None:
        self._retriever = retriever
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._index_id = index_id

    def _path_for_key(self, key: str) -> Path:
        return self._cache_dir / f"{key}.json"

    def retrieve(self, query: str, top_k: int) -> List[Passage]:
        key = compute_retrieval_cache_key(query, top_k, self._index_id)
        path = self._path_for_key(key)

        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return [Passage(**item) for item in data]

        passages = self._retriever.retrieve(query, top_k)
        path.write_text(
            json.dumps([asdict(passage) for passage in passages], ensure_ascii=True),
            encoding="utf-8",
        )
        return passages
