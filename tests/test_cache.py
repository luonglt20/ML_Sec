from medqa_multiagent.cache import OnDiskLLMCache

from tests.fakes import FakeLLMClient



def test_cache_miss_delegates_to_wrapped_client(tmp_path):
    fake = FakeLLMClient(["Final Answer: A"])
    cache = OnDiskLLMCache(fake, tmp_path)

    response = cache.complete(role="direct", prompt="Q?", model="gpt-4o-mini", temperature=0.0)

    assert response.text == "Final Answer: A"
    assert response.cache_hit is False
    assert len(fake.calls) == 1


def test_identical_request_is_a_cache_hit_and_avoids_a_second_call(tmp_path):
    fake = FakeLLMClient(["Final Answer: A"])
    cache = OnDiskLLMCache(fake, tmp_path)

    first = cache.complete(role="direct", prompt="Q?", model="gpt-4o-mini", temperature=0.0)
    second = cache.complete(role="direct", prompt="Q?", model="gpt-4o-mini", temperature=0.0)

    assert len(fake.calls) == 1  # only the first call reached the wrapped client
    assert first.text == second.text
    assert second.cache_hit is True


def test_differing_prompt_is_a_cache_miss(tmp_path):
    fake = FakeLLMClient(["Final Answer: A", "Final Answer: B"])
    cache = OnDiskLLMCache(fake, tmp_path)

    cache.complete(role="direct", prompt="Q1?", model="gpt-4o-mini", temperature=0.0)
    cache.complete(role="direct", prompt="Q2?", model="gpt-4o-mini", temperature=0.0)

    assert len(fake.calls) == 2


def test_cache_persists_across_separate_cache_instances(tmp_path):
    fake_first_run = FakeLLMClient(["Final Answer: A"])
    cache_first_run = OnDiskLLMCache(fake_first_run, tmp_path)
    first = cache_first_run.complete(
        role="direct", prompt="Q?", model="gpt-4o-mini", temperature=0.0
    )

    # Simulate a fresh process re-run against the same cache directory: the
    # wrapped client here would raise if actually called, proving the
    # re-run spends no new LLM calls.
    fake_second_run = FakeLLMClient([])
    cache_second_run = OnDiskLLMCache(fake_second_run, tmp_path)
    second = cache_second_run.complete(
        role="direct", prompt="Q?", model="gpt-4o-mini", temperature=0.0
    )

    assert len(fake_second_run.calls) == 0
    assert second.cache_hit is True
    assert second.text == first.text
