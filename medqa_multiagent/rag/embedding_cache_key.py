"""Cache-key derivation for the on-disk embedding cache.

Mirrors `medqa_multiagent.cache_key.compute_cache_key`'s discipline: a
stable, content-addressed hash of exactly the parameters that determine an
embedding call's output -- which encoder ("query" vs "passage", since
MedCPT's two encoders are asymmetric, see `embeddings.py`), the exact text
embedded, and the model identifier. Two requests identical in all three
always produce the same key; changing any one of them changes it.
"""

from __future__ import annotations

import hashlib
import json


def compute_embedding_cache_key(kind: str, text: str, model_name: str) -> str:
    """Derive a stable, content-addressed cache key for one embedding call.

    Args:
        kind: Which encoder produced (or would produce) the embedding --
            `"query"` or `"passage"`. MedCPT's query/article encoders are
            different models, so the same text embedded as a query vs. a
            passage must never share a cache entry.
        text: The exact text embedded.
        model_name: The embedding model identifier (e.g.
            `"ncbi/MedCPT-Query-Encoder"`).

    Returns:
        A 64-character hex SHA-256 digest of a canonical encoding of the
        above parameters.
    """
    payload = {"kind": kind, "text": text, "model_name": model_name}
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
