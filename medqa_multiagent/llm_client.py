"""The single LLM client interface used by every agent role.

Abstracts the underlying provider behind one `LLMClient` interface. Two
concrete configurations exist: a development configuration (`gpt-4o-mini`,
OpenAI) and a reported-evaluation configuration (`deepseek-chat`, DeepSeek).
Switching between them is a matter of which `model` a `RunConfig` names --
`create_llm_client` is the only place that maps a model identifier to a
provider, so no agent-role code ever needs to change to switch phases.

Every call runs at whatever temperature the caller passes (the project
convention is 0, enforced by `RunConfig`, not by this module) and surfaces
per-call token usage and wall-clock latency via `LLMResponse`.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional, Protocol


@dataclass(frozen=True)
class LLMResponse:
    """The result of one LLM call, with everything needed to trace/cost it.

    Attributes:
        text: The raw model output text.
        model: The model identifier used for this call.
        temperature: The sampling temperature used for this call.
        prompt: The exact, fully-rendered prompt sent to the model.
        role: The agent role that issued this call (e.g. "direct", "router").
        prompt_tokens: Number of prompt/input tokens billed for this call.
        completion_tokens: Number of completion/output tokens billed for this call.
        total_tokens: Total tokens billed for this call.
        latency_seconds: Wall-clock time spent on this call.
        retrieved_context_id: Identifier of the retrieved context used to
            build `prompt`, if any (used for cache-key derivation; `None`
            when retrieval isn't involved, e.g. every V0 call).
        cache_hit: Whether this response was served from the on-disk cache
            rather than a fresh provider call. Always `False` for responses
            returned directly by an `LLMClient` that isn't cache-wrapped.
    """

    text: str
    model: str
    temperature: float
    prompt: str
    role: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_seconds: float
    retrieved_context_id: Optional[str] = None
    cache_hit: bool = False


class LLMClient(Protocol):
    """The one interface every agent role calls through, regardless of provider."""

    def complete(
        self,
        role: str,
        prompt: str,
        model: str,
        temperature: float,
        retrieved_context_id: Optional[str] = None,
    ) -> LLMResponse: ...


@dataclass(frozen=True)
class ProviderConfig:
    """Where/how to reach a given model's provider."""

    base_url: str
    api_key_env_var: str


#: Maps a model identifier (as named by `RunConfig.model`) to the provider
#: that serves it. This is the *only* place model->provider selection lives;
#: adding/repointing a model is a registry edit here, never agent-role code.
_PROVIDER_REGISTRY = {
    "gpt-4o-mini": ProviderConfig(
        base_url="https://api.openai.com/v1", api_key_env_var="OPENAI_API_KEY"
    ),
    "deepseek-chat": ProviderConfig(
        base_url="https://api.deepseek.com/v1", api_key_env_var="DEEPSEEK_API_KEY"
    ),
}


class UnknownModelError(ValueError):
    """Raised when `create_llm_client` is asked for a model with no registered provider."""


class OpenAICompatibleClient:
    """`LLMClient` for any OpenAI Chat-Completions-compatible provider.

    Both of this project's two concrete configurations (OpenAI's
    `gpt-4o-mini` and DeepSeek's `deepseek-chat`) speak this same REST API
    shape, so one implementation serves both -- only `base_url`/`api_key`
    differ, which `create_llm_client` supplies from the provider registry.
    """

    def __init__(
        self, base_url: str, api_key: str, timeout: float | None = None
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = (
            timeout
            if timeout is not None
            else float(os.environ.get("MEDQA_LLM_TIMEOUT_SECONDS", "180"))
        )
        self._max_retries = int(os.environ.get("MEDQA_LLM_MAX_RETRIES", "4"))
        if self._timeout <= 0:
            raise ValueError("LLM request timeout must be positive")
        if self._max_retries < 0:
            raise ValueError("MEDQA_LLM_MAX_RETRIES must be non-negative")

    @staticmethod
    def _retry_delay(attempt: int, retry_after: str | None = None) -> float:
        """Return a bounded exponential-backoff delay, honoring Retry-After."""
        if retry_after:
            try:
                return min(float(retry_after), 60.0)
            except ValueError:
                pass
        return min(2.0**attempt, 30.0)

    def complete(
        self,
        role: str,
        prompt: str,
        model: str,
        temperature: float,
        retrieved_context_id: Optional[str] = None,
    ) -> LLMResponse:
        payload = {
            "model": model,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        request = urllib.request.Request(
            f"{self._base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )

        start = time.monotonic()
        for attempt in range(self._max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self._timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                retryable = exc.code == 429 or 500 <= exc.code < 600
                if not retryable or attempt == self._max_retries:
                    raise RuntimeError(
                        f"LLM provider request failed ({exc.code}): {detail}"
                    ) from exc
                delay = self._retry_delay(attempt, exc.headers.get("Retry-After"))
                logging.warning(
                    "LLM request returned HTTP %s; retrying in %.1fs (%s/%s)",
                    exc.code,
                    delay,
                    attempt + 1,
                    self._max_retries,
                )
            except (TimeoutError, socket.timeout, urllib.error.URLError) as exc:
                if attempt == self._max_retries:
                    raise RuntimeError(
                        "LLM provider request timed out or could not connect after "
                        f"{self._max_retries + 1} attempt(s)"
                    ) from exc
                delay = self._retry_delay(attempt)
                logging.warning(
                    "LLM request failed with %s; retrying in %.1fs (%s/%s)",
                    type(exc).__name__,
                    delay,
                    attempt + 1,
                    self._max_retries,
                )
            time.sleep(delay)
        latency = time.monotonic() - start

        text = body["choices"][0]["message"]["content"]
        usage = body.get("usage", {})
        return LLMResponse(
            text=text,
            model=model,
            temperature=temperature,
            prompt=prompt,
            role=role,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            latency_seconds=latency,
            retrieved_context_id=retrieved_context_id,
        )


def create_llm_client(model: str) -> LLMClient:
    """Build the `LLMClient` for `model`, via the provider registry.

    Selecting between the development configuration (`gpt-4o-mini`, OpenAI)
    and the reported-evaluation configuration (`deepseek-chat`, DeepSeek) is
    purely a matter of which `model` a `RunConfig` names -- no code change
    required to switch.

    Raises:
        UnknownModelError: if `model` has no registered provider.
        RuntimeError: if the provider's required API key environment
            variable isn't set.
    """
    provider = _PROVIDER_REGISTRY.get(model)
    if provider is None:
        raise UnknownModelError(
            f"No provider registered for model {model!r}. "
            f"Known models: {sorted(_PROVIDER_REGISTRY)}"
        )

    api_key = os.environ.get(provider.api_key_env_var)
    if not api_key:
        raise RuntimeError(
            f"Environment variable {provider.api_key_env_var} is not set; "
            f"required to call model {model!r}."
        )
    return OpenAICompatibleClient(base_url=provider.base_url, api_key=api_key)


class LoggingLLMClient:
    """Decorator: logs every call's model, temperature, and rendered prompt.

    Wraps any `LLMClient` -- typically the outermost layer, above caching --
    so every call this pipeline makes, cache hit or miss, is logged, and any
    past prediction can be traced back to precisely what produced it. Logs
    one JSON-encoded record per call at INFO level.
    """

    def __init__(self, client: LLMClient, logger: Optional[logging.Logger] = None) -> None:
        self._client = client
        self._logger = logger or logging.getLogger(f"{__name__}.calls")

    def complete(
        self,
        role: str,
        prompt: str,
        model: str,
        temperature: float,
        retrieved_context_id: Optional[str] = None,
    ) -> LLMResponse:
        response = self._client.complete(
            role=role,
            prompt=prompt,
            model=model,
            temperature=temperature,
            retrieved_context_id=retrieved_context_id,
        )
        self._logger.info(
            json.dumps(
                {
                    "role": role,
                    "model": model,
                    "temperature": temperature,
                    "prompt": prompt,
                    "retrieved_context_id": retrieved_context_id,
                    "cache_hit": response.cache_hit,
                    "prompt_tokens": response.prompt_tokens,
                    "completion_tokens": response.completion_tokens,
                    "total_tokens": response.total_tokens,
                    "latency_seconds": response.latency_seconds,
                },
                ensure_ascii=True,
            )
        )
        return response
