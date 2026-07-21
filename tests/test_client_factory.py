import logging

from medqa_multiagent import client_factory
from medqa_multiagent.config import RunConfig

from tests.fakes import FakeLLMClient



def make_config(**overrides):
    data = dict(
        model="gpt-4o-mini",
        temperature=0.0,
        dev_sample_size=3,
        official_test_sample_size=2,
        seed=42,
        rag_top_k=3,
        rag_chunk_size=256,
        memory_top_k=3,
    )
    data.update(overrides)
    return RunConfig.from_mapping(data)


def test_build_llm_client_caches_and_logs_calls(tmp_path, monkeypatch, caplog):
    fake = FakeLLMClient(["Final Answer: A"])
    monkeypatch.setattr(client_factory, "create_llm_client", lambda model: fake)
    config = make_config()

    client = client_factory.build_llm_client(config, tmp_path / "cache")

    with caplog.at_level(logging.INFO):
        first = client.complete(role="direct", prompt="Q?", model=config.model, temperature=0.0)
        second = client.complete(role="direct", prompt="Q?", model=config.model, temperature=0.0)

    assert first.text == "Final Answer: A"
    assert second.text == "Final Answer: A"
    assert len(fake.calls) == 1  # the second call was a cache hit
    assert len(caplog.records) == 2  # both calls (hit and miss) are logged


def test_build_llm_client_uses_default_cache_dir_when_not_given(tmp_path, monkeypatch):
    fake = FakeLLMClient(["Final Answer: A"])
    monkeypatch.setattr(client_factory, "create_llm_client", lambda model: fake)
    monkeypatch.chdir(tmp_path)
    config = make_config()

    client_factory.build_llm_client(config)

    assert (tmp_path / client_factory.DEFAULT_CACHE_DIR).is_dir()
