"""Cache-key derivation for the on-disk retrieval cache.

Mirrors `medqa_multiagent.cache_key.compute_cache_key`'s discipline: a
stable, content-addressed hash of exactly the parameters that determine a
retrieval call's output -- the exact query text, `top_k`, and an
`index_id` identifying which built index is being queried (so a cache
built against one corpus/index version is never silently reused against a
different one).
"""

from __future__ import annotations

import hashlib
import json


def compute_retrieval_cache_key(query: str, top_k: int, index_id: str) -> str:
    """Derive a stable, content-addressed cache key for one retrieval call.

    Args:
        query: The exact query text retrieved against.
        top_k: The number of passages requested.
        index_id: A stable identifier for the index/corpus version queried
            (e.g. its on-disk directory path) -- distinct index builds
            must never collide in the cache.

    Returns:
        A 64-character hex SHA-256 digest of a canonical encoding of the
        above parameters.
    """
    payload = {"query": query, "top_k": int(top_k), "index_id": index_id}
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
