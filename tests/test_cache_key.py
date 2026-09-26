from typing import Any, Dict

from medqa_multiagent.cache_key import compute_cache_key


def base_kwargs(**overrides: Any) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = dict(
        role="reasoner",
        prompt="Question: ...\nOptions: ...",
        model="gpt-4o-mini",
        temperature=0.0,
        retrieved_context_id="ctx-abc123",
    )
    kwargs.update(overrides)
    return kwargs


def test_identical_requests_produce_the_same_key():
    key_a = compute_cache_key(**base_kwargs())
    key_b = compute_cache_key(**base_kwargs())
    assert key_a == key_b


def test_key_is_a_sha256_hex_digest():
    key = compute_cache_key(**base_kwargs())
    assert isinstance(key, str)
    assert len(key) == 64
    assert all(c in "0123456789abcdef" for c in key)


def test_changing_role_changes_the_key():
    key_a = compute_cache_key(**base_kwargs(role="reasoner"))
    key_b = compute_cache_key(**base_kwargs(role="verifier"))
    assert key_a != key_b


def test_changing_prompt_changes_the_key():
    key_a = compute_cache_key(**base_kwargs(prompt="prompt A"))
    key_b = compute_cache_key(**base_kwargs(prompt="prompt B"))
    assert key_a != key_b


def test_changing_model_changes_the_key():
    key_a = compute_cache_key(**base_kwargs(model="gpt-4o-mini"))
    key_b = compute_cache_key(**base_kwargs(model="deepseek-chat"))
    assert key_a != key_b


def test_changing_temperature_changes_the_key():
    key_a = compute_cache_key(**base_kwargs(temperature=0.0))
    key_b = compute_cache_key(**base_kwargs(temperature=0.7))
    assert key_a != key_b


def test_changing_retrieved_context_id_changes_the_key():
    key_a = compute_cache_key(**base_kwargs(retrieved_context_id="ctx-1"))
    key_b = compute_cache_key(**base_kwargs(retrieved_context_id="ctx-2"))
    assert key_a != key_b


def test_no_retrieved_context_id_is_distinct_from_any_real_id():
    key_none = compute_cache_key(**base_kwargs(retrieved_context_id=None))
    key_some = compute_cache_key(**base_kwargs(retrieved_context_id="ctx-1"))
    assert key_none != key_some


def test_retrieved_context_id_defaults_to_none():
    kwargs = base_kwargs()
    del kwargs["retrieved_context_id"]
    key_default = compute_cache_key(**kwargs)
    key_explicit_none = compute_cache_key(**base_kwargs(retrieved_context_id=None))
    assert key_default == key_explicit_none
