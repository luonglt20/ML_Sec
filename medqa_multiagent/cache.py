"""On-disk cache wrapping the LLM client interface.

Keyed via `cache_key.compute_cache_key` (agent role, fully-rendered prompt,
model identifier, temperature, retrieved-context identifier), so identical
requests never re-spend API cost, and any single differing parameter is a
guaranteed miss -- safe under temperature=0, where a hit requires
bit-for-bit identical inputs.

This module contains no LLM-calling logic itself; it only decides whether a
given request has been seen before and, if not, delegates to the wrapped
`LLMClient` and persists the result.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Union

from .cache_key import compute_cache_key
from .llm_client import LLMClient, LLMResponse


class OnDiskLLMCache:
    """`LLMClient` decorator backed by one JSON file per cache key on disk.

    A cache hit occurs only for a request whose (role, prompt, model,
    temperature, retrieved_context_id) tuple exactly matches a previously
    seen request -- reusing `cache_key.compute_cache_key`'s content-addressed
    derivation. On a hit, the wrapped client is never called, so re-running
    an evaluation after a non-prompt code change doesn't re-spend API cost
    regenerating identical responses. Cache entries persist across processes
    and days, since they're plain files under `cache_dir`.
    """

    def __init__(self, client: LLMClient, cache_dir: Union[str, Path]) -> None:
        self._client = client
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _path_for_key(self, key: str) -> Path:
        return self._cache_dir / f"{key}.json"

    def complete(
        self,
        role: str,
        prompt: str,
        model: str,
        temperature: float,
        retrieved_context_id: Optional[str] = None,
    ) -> LLMResponse:
        key = compute_cache_key(
            role=role,
            prompt=prompt,
            model=model,
            temperature=temperature,
            retrieved_context_id=retrieved_context_id,
        )
        path = self._path_for_key(key)

        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            data["cache_hit"] = True
            return LLMResponse(**data)

        response = self._client.complete(
            role=role,
            prompt=prompt,
            model=model,
            temperature=temperature,
            retrieved_context_id=retrieved_context_id,
        )
        path.write_text(json.dumps(asdict(response), ensure_ascii=True), encoding="utf-8")
        return response
