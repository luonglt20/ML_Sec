"""On-disk cache wrapping the embedding client interface.

Mirrors `medqa_multiagent.cache.OnDiskLLMCache`: keyed via
`embedding_cache_key.compute_embedding_cache_key` (encoder kind, exact
text, model identifier), so re-embedding identical text never re-spends
compute, and cache hits are per-text -- a batched `embed_passages` call
with some already-cached texts only computes the uncached ones, then
reassembles the full result in the original order.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence, Union

from .embedding_cache_key import compute_embedding_cache_key
from .embeddings import MEDCPT_ARTICLE_MODEL, MEDCPT_QUERY_MODEL, EmbeddingClient


class OnDiskEmbeddingCache:
    """`EmbeddingClient` decorator backed by one JSON file per cache key.

    Args:
        client: The `EmbeddingClient` to wrap.
        cache_dir: Directory to persist cache entries under.
        query_model_name: Identifier recorded for query-embedding cache
            keys (default: the MedCPT query encoder's identifier).
        passage_model_name: Identifier recorded for passage-embedding
            cache keys (default: the MedCPT article encoder's identifier).
    """

    def __init__(
        self,
        client: EmbeddingClient,
        cache_dir: Union[str, Path],
        query_model_name: str = MEDCPT_QUERY_MODEL,
        passage_model_name: str = MEDCPT_ARTICLE_MODEL,
    ) -> None:
        self._client = client
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._query_model_name = query_model_name
        self._passage_model_name = passage_model_name

    def _path_for_key(self, key: str) -> Path:
        return self._cache_dir / f"{key}.json"

    def _read(self, key: str) -> Union[List[float], None]:
        path = self._path_for_key(key)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _write(self, key: str, vector: List[float]) -> None:
        path = self._path_for_key(key)
        path.write_text(json.dumps(vector), encoding="utf-8")

    def embed_query(self, text: str) -> List[float]:
        key = compute_embedding_cache_key("query", text, self._query_model_name)
        cached = self._read(key)
        if cached is not None:
            return cached
        vector = self._client.embed_query(text)
        self._write(key, vector)
        return vector

    def embed_passages(self, texts: Sequence[str]) -> List[List[float]]:
        keys = [
            compute_embedding_cache_key("passage", text, self._passage_model_name)
            for text in texts
        ]
        cached = [self._read(key) for key in keys]

        uncached_indices = [i for i, vector in enumerate(cached) if vector is None]
        if uncached_indices:
            fresh = self._client.embed_passages([texts[i] for i in uncached_indices])
            for position, index in enumerate(uncached_indices):
                cached[index] = fresh[position]
                self._write(keys[index], fresh[position])

        return cached  # type: ignore[return-value]
