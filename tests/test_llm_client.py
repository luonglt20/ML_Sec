import json
from typing import cast
from unittest.mock import MagicMock, patch

import pytest

from medqa_multiagent.llm_client import (
    LoggingLLMClient,
    OpenAICompatibleClient,
    UnknownModelError,
    create_llm_client,
)

from fakes import FakeLLMClient


def test_create_llm_client_unknown_model_raises(monkeypatch):
    with pytest.raises(UnknownModelError):
        create_llm_client("not-a-real-model")


def test_create_llm_client_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        create_llm_client("gpt-4o-mini")


def test_create_llm_client_builds_client_with_correct_provider(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    client = create_llm_client("gpt-4o-mini")
    assert isinstance(client, OpenAICompatibleClient)
    openai_client = cast(OpenAICompatibleClient, client)
    assert openai_client._base_url == "https://api.openai.com/v1"
    assert openai_client._api_key == "sk-test"


def test_create_llm_client_deepseek_uses_deepseek_env_var(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-test")
    client = create_llm_client("deepseek-chat")
    assert isinstance(client, OpenAICompatibleClient)
    deepseek_client = cast(OpenAICompatibleClient, client)
    assert deepseek_client._base_url == "https://api.deepseek.com/v1"
    assert deepseek_client._api_key == "ds-test"


def _fake_http_response(body: dict):
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps(body).encode("utf-8")
    mock_response.__enter__.return_value = mock_response
    mock_response.__exit__.return_value = False
    return mock_response


def test_openai_compatible_client_parses_response():
    body = {
        "choices": [{"message": {"content": "Final Answer: A"}}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
    }
    client = OpenAICompatibleClient(base_url="https://example.com/v1", api_key="key")

    with patch("medqa_multiagent.llm_client.urllib.request.urlopen", return_value=_fake_http_response(body)):
        response = client.complete(
            role="direct", prompt="hello", model="gpt-4o-mini", temperature=0.0
        )

    assert response.text == "Final Answer: A"
    assert response.prompt_tokens == 12
    assert response.completion_tokens == 3
    assert response.total_tokens == 15
    assert response.model == "gpt-4o-mini"
    assert response.temperature == 0.0
    assert response.prompt == "hello"
    assert response.role == "direct"
    assert response.cache_hit is False


def test_logging_llm_client_delegates_and_logs(caplog):
    client = FakeLLMClient(["Final Answer: A"])
    logging_client = LoggingLLMClient(client)

    with caplog.at_level("INFO"):
        response = logging_client.complete(
            role="direct", prompt="What is 2+2?", model="gpt-4o-mini", temperature=0.0
        )

    assert response.text == "Final Answer: A"
    assert len(client.calls) == 1

    assert len(caplog.records) == 1
    logged = json.loads(caplog.records[0].message)
    assert logged["role"] == "direct"
    assert logged["model"] == "gpt-4o-mini"
    assert logged["temperature"] == 0.0
    assert logged["prompt"] == "What is 2+2?"
