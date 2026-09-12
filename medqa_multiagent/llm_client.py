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

    def __init__(self, base_url: str, api_key: str, timeout: float = 60.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout

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
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"LLM provider request failed ({exc.code}): {detail}"
            ) from exc
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


class OllamaClient:
    """Local Ollama chat client for reproducible real-LLM evaluation.

    ``ollama/<model>`` identifiers are handled by ``create_llm_client``. The
    client has no API key and reports Ollama's actual prompt/evaluation token
    counts and generation duration. It can optionally render a structured
    frontend query for the explicitly-labelled frontend-only ablation; this
    does not turn an arbitrary Ollama model into StruQ.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        timeout: float = 300.0,
        use_structured_queries: bool = False,
        max_new_tokens: int = 256,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self.use_structured_queries = use_structured_queries
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        self.max_new_tokens = max_new_tokens
        self.sanitization_events = 0
        self.structured_queries = 0

    @staticmethod
    def _ollama_model_name(model: str) -> str:
        return model.split("/", 1)[1] if model.startswith("ollama/") else model

    def _complete_prompt(
        self,
        role: str,
        prompt: str,
        model: str,
        temperature: float,
        retrieved_context_id: Optional[str],
    ) -> LLMResponse:
        payload = {
            "model": self._ollama_model_name(model),
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": self.max_new_tokens,
            },
        }
        request = urllib.request.Request(
            f"{self._base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        start = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                "Ollama is unavailable. Start it with `ollama serve` and verify "
                "that the selected model is installed."
            ) from exc
        latency = time.monotonic() - start
        text = str(body.get("message", {}).get("content", ""))
        prompt_tokens = int(body.get("prompt_eval_count", 0))
        completion_tokens = int(body.get("eval_count", 0))
        return LLMResponse(
            text=text,
            model=f"ollama/{self._ollama_model_name(model)}",
            temperature=temperature,
            prompt=prompt,
            role=role,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_seconds=latency,
            retrieved_context_id=retrieved_context_id,
        )

    def complete(
        self,
        role: str,
        prompt: str,
        model: str,
        temperature: float,
        retrieved_context_id: Optional[str] = None,
    ) -> LLMResponse:
        if self.use_structured_queries:
            from .security.structured import StructuredQuery, compose_struq_prompt

            prompt = compose_struq_prompt(
                StructuredQuery(instruction=prompt, data=""), apply_filter=True
            )
        return self._complete_prompt(
            role, prompt, model, temperature, retrieved_context_id
        )

    def complete_structured(
        self,
        role: str,
        query: object,
        model: str,
        temperature: float,
        retrieved_context_id: Optional[str] = None,
    ) -> LLMResponse:
        from .security.structured import StructuredQuery, compose_struq_prompt, recursive_filter

        if not isinstance(query, StructuredQuery):
            raise TypeError("query must be a StructuredQuery")
        self.structured_queries += 1
        if recursive_filter(query.data) != query.data:
            self.sanitization_events += 1
        prompt = compose_struq_prompt(query, apply_filter=True)
        return self._complete_prompt(
            role, prompt, model, temperature, retrieved_context_id
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
    if model.startswith("ollama/"):
        return OllamaClient(base_url=os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"))

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
