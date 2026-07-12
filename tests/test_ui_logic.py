import json

from medqa_multiagent import client_factory
from medqa_multiagent.config import RunConfig
from medqa_multiagent.ui import logic

from fakes import FakeLLMClient


def make_config_file(path, **overrides):
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
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_load_config_returns_run_config_on_success(tmp_path):
    config_path = make_config_file(tmp_path / "config.json")

    config, error = logic.load_config(config_path)

    assert error is None
    assert isinstance(config, RunConfig)
    assert config.model == "gpt-4o-mini"


def test_load_config_returns_readable_error_for_missing_file(tmp_path):
    config, error = logic.load_config(tmp_path / "does-not-exist.json")

    assert config is None
    assert error is not None
    assert "does-not-exist.json" in error


def test_load_config_returns_readable_error_for_invalid_config(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"model": "gpt-4o-mini"}), encoding="utf-8")

    config, error = logic.load_config(config_path)

    assert config is None
    assert error is not None


def test_get_answer_returns_result_on_success(tmp_path, monkeypatch):
    config_path = make_config_file(tmp_path / "config.json")
    config, _ = logic.load_config(config_path)
    assert config is not None
    fake = FakeLLMClient(["Some reasoning.\nFinal Answer: B"])
    monkeypatch.setattr(client_factory, "create_llm_client", lambda model: fake)

    outcome = logic.get_answer(
        "Q?", {"A": "x", "B": "y"}, "V0", config, tmp_path / "cache"
    )

    assert outcome.error is None
    assert outcome.result is not None
    assert outcome.result.answer == "B"
    assert outcome.result.explanation == "Some reasoning."
    assert outcome.result.is_valid is True


def test_get_answer_reports_missing_api_key_as_a_readable_error(tmp_path, monkeypatch):
    config_path = make_config_file(tmp_path / "config.json")
    config, _ = logic.load_config(config_path)
    assert config is not None
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    outcome = logic.get_answer(
        "Q?", {"A": "x", "B": "y"}, "V0", config, tmp_path / "cache"
    )

    assert outcome.result is None
    assert outcome.error is not None
    assert "OPENAI_API_KEY" in outcome.error


def test_get_answer_reports_unsupported_variant_as_a_readable_error(tmp_path, monkeypatch):
    config_path = make_config_file(tmp_path / "config.json")
    config, _ = logic.load_config(config_path)
    assert config is not None
    fake = FakeLLMClient([])
    monkeypatch.setattr(client_factory, "create_llm_client", lambda model: fake)

    outcome = logic.get_answer(
        "Q?", {"A": "x", "B": "y"}, "V1", config, tmp_path / "cache"
    )

    assert outcome.result is None
    assert outcome.error is not None
    assert "V1" in outcome.error


def test_get_answer_resubmission_is_a_cache_hit_with_identical_result(tmp_path, monkeypatch):
    config_path = make_config_file(tmp_path / "config.json")
    config, _ = logic.load_config(config_path)
    assert config is not None
    cache_dir = tmp_path / "cache"

    first_client = FakeLLMClient(["Final Answer: A"])
    monkeypatch.setattr(client_factory, "create_llm_client", lambda model: first_client)
    first_outcome = logic.get_answer("Q?", {"A": "x", "B": "y"}, "V0", config, cache_dir)

    # A fresh client with *no* scripted responses -- if the resubmission
    # makes any real call, FakeLLMClient raises, proving it's a cache hit.
    second_client = FakeLLMClient([])
    monkeypatch.setattr(client_factory, "create_llm_client", lambda model: second_client)
    second_outcome = logic.get_answer("Q?", {"A": "x", "B": "y"}, "V0", config, cache_dir)

    assert second_outcome.error is None
    assert len(second_client.calls) == 0
    assert first_outcome.result is not None
    assert second_outcome.result is not None
    assert second_outcome.result.answer == first_outcome.result.answer
    assert second_outcome.result.explanation == first_outcome.result.explanation
