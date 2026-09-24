import json

import pytest

from medqa_multiagent.llm_client import LLMResponse
from medqa_multiagent.semantic_defense import (
    SEMANTIC_GUARD_SYSTEM,
    RoleSeparatedGuardClient,
    parse_guard_response,
)


def _response(text):
    return LLMResponse(
        text=text,
        model="deepseek-chat",
        temperature=0,
        prompt="Clinical stem",
        role="security_guard",
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        latency_seconds=0.5,
    )


def test_parse_guard_response_preserves_only_question_field():
    result = parse_guard_response("Clinical stem", _response('{"question":"Clinical stem"}'))
    assert result.question == "Clinical stem"
    assert result.total_tokens == 15
    with pytest.raises(ValueError):
        parse_guard_response("Clinical stem", _response('{"answer":"B"}'))
    with pytest.raises(ValueError):
        parse_guard_response("Clinical stem", _response('{"question":""}'))


def test_guard_policy_uses_system_role_and_raw_question_uses_user_role(monkeypatch):
    sent = {}

    class FakeHTTPResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return json.dumps({
                "choices": [{"message": {"content": '{"question":"Clinical stem"}'}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }).encode()

    def fake_urlopen(request, timeout):
        sent.update(json.loads(request.data))
        return FakeHTTPResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = RoleSeparatedGuardClient(api_key="test-only")
    client.complete("security_guard", "Clinical stem", "deepseek-chat", 0.0)
    assert sent["messages"] == [
        {"role": "system", "content": SEMANTIC_GUARD_SYSTEM},
        {"role": "user", "content": "Clinical stem"},
    ]
