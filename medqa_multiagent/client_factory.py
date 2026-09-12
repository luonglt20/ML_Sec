"""Builds the one standard LLM client stack shared by every caller.

Both the CLI (`cli.py`) and the Streamlit demo UI (`ui/`) need the exact
same client stack: an on-disk cache (so identical requests never re-spend
API cost) wrapped in a call logger (so every call's model/temperature/
prompt is logged), around the provider client selected by `RunConfig.model`.
Building it in exactly one place keeps those callers from silently drifting
apart, and keeps the on-disk cache directly reusable across both -- e.g. a
question already answered via the CLI is a cache hit when re-submitted
through the UI, and vice versa.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

from .cache import OnDiskLLMCache
from .config import RunConfig
from .llm_client import LLMClient, LoggingLLMClient, create_llm_client

#: Default on-disk cache directory, shared by every caller unless overridden.
DEFAULT_CACHE_DIR = ".cache/llm"


def _build_provider_client(config: RunConfig) -> LLMClient:
    """Create the provider client selected by a run configuration."""
    if config.model in ("unified", "auto", "groq", "gemini", "deepseek"):
        from .unified_llm_client import UnifiedLLMClient
        return UnifiedLLMClient()
    return create_llm_client(config.model)


def build_uncached_llm_client(config: RunConfig) -> LLMClient:
    """Build a logging client that sends every call to the provider.

    Security evaluation must measure each clean/attack/defense condition as a
    distinct real model invocation. It therefore cannot use the normal prompt
    cache, because a sanitizer can deliberately make an attacked prompt render
    identically to a clean one.
    """
    return LoggingLLMClient(_build_provider_client(config))


def build_llm_client(
    config: RunConfig, cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR
) -> LLMClient:
    """Build the standard cache- and logging-wrapped `LLMClient` for `config.model`."""
    inner = _build_provider_client(config)

    cached = OnDiskLLMCache(inner, cache_dir)
    return LoggingLLMClient(cached)
