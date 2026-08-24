"""Cache-key derivation for the future on-disk LLM/embedding cache.

A cache key is a stable, content-addressed hash of exactly the parameters
that determine an LLM call's output under temperature=0: agent role, the
fully-rendered prompt, model identifier, temperature, and (where
applicable) a retrieved-context identifier. Two requests identical in
every one of these parameters must produce the same key; changing any
single one of them must change it. This module contains no caching
storage itself -- only the pure key-derivation function a future on-disk
cache will use.
"""

from __future__ import annotations

import hashlib
import json
from typing import Optional


def compute_cache_key(
    role: str,
    prompt: str,
    model: str,
    temperature: float,
    retrieved_context_id: Optional[str] = None,
) -> str:
    """Derive a stable, content-addressed cache key for one LLM request.

    Args:
        role: The agent role issuing the request (e.g. "router", "reasoner",
            "verifier", "direct").
        prompt: The fully-rendered prompt text sent to the model.
        model: The model identifier (e.g. "gpt-4o-mini", "deepseek-chat").
        temperature: The sampling temperature used for the request.
        retrieved_context_id: A stable identifier for the retrieved context
            used to build the prompt (e.g. a hash of the retrieved passages),
            when retrieval is involved. `None` when not applicable.

    Returns:
        A 64-character hex SHA-256 digest of a canonical encoding of the
        above parameters. Bit-for-bit identical inputs always produce the
        same key; any single differing parameter produces a different key.
    """
    payload = {
        "role": role,
        "prompt": prompt,
        "model": model,
        "temperature": float(temperature),
        "retrieved_context_id": retrieved_context_id,
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
