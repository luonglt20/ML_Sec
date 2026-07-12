import json

from medqa_multiagent import client_factory
from medqa_multiagent.config import RunConfig
from medqa_multiagent.rag.retriever import Passage
from medqa_multiagent.ui import logic

from fakes import FakeLLMClient, FakeRetriever


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

    # V2 (not V1): V1 (RAG-only) is now implemented (see #3), so an
    # "unsupported variant" test needs a variant still awaiting its own
    # ticket.
    outcome = logic.get_answer(
        "Q?", {"A": "x", "B": "y"}, "V2", config, tmp_path / "cache"
    )

    assert outcome.result is None
    assert outcome.error is not None
    assert "V2" in outcome.error


def test_get_answer_works_for_v1_with_no_ui_code_change(tmp_path, monkeypatch):
    """Confirms #3's awareness check: the UI (`ui/logic.get_answer`) needs
    no code change for V1 -- `SUPPORTED_VARIANTS` already includes it, and
    `answer_question` builds a default retriever on its own when the UI
    (which never passes one) doesn't supply it."""
    config_path = make_config_file(tmp_path / "config.json")
    config, _ = logic.load_config(config_path)
    assert config is not None
    fake_llm = FakeLLMClient(["Some reasoning.\nFinal Answer: B"])
    monkeypatch.setattr(client_factory, "create_llm_client", lambda model: fake_llm)
    fake_retriever = FakeRetriever(
        [Passage(passage_id="p1", source="BookA", text="relevant text", score=0.9)]
    )
    monkeypatch.setattr(
        "medqa_multiagent.entrypoint._default_retriever", lambda config: fake_retriever
    )

    outcome = logic.get_answer("Q?", {"A": "x", "B": "y"}, "V1", config, tmp_path / "cache")

    assert outcome.error is None
    assert outcome.result is not None
    assert outcome.result.variant == "V1"
    assert outcome.result.answer == "B"
    # The trace has content the UI's existing generic `render_trace` can
    # display (a dict whose values are lists/dicts render via `st.json`).
    assert "retrieved_passages" in outcome.result.trace


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
