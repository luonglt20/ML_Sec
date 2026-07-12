import json

import pytest

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

    # V3 (not V2): V2 (multi-agent Router->Reasoner->Verifier) is now
    # implemented (see #4), so an "unsupported variant" test needs a
    # variant still awaiting its own ticket.
    outcome = logic.get_answer(
        "Q?", {"A": "x", "B": "y"}, "V3", config, tmp_path / "cache"
    )

    assert outcome.result is None
    assert outcome.error is not None
    assert "V3" in outcome.error


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


def test_get_answer_works_for_v2_with_no_ui_code_change(tmp_path, monkeypatch):
    """Confirms #4's awareness check: the UI (`ui/logic.get_answer`) needs
    no code change for V2 -- `SUPPORTED_VARIANTS` already includes it, and
    `answer_question` builds a default retriever on its own when the UI
    (which never passes one) doesn't supply it. The trace's nested dicts
    (`reasoner_candidate`, `verifier_decision`) render fine through the
    UI's existing generic `render_trace`/`_render_trace_value` (dict/list
    values go through `st.json`, everything else through `st.write`)."""
    config_path = make_config_file(tmp_path / "config.json")
    config, _ = logic.load_config(config_path)
    assert config is not None
    fake_llm = FakeLLMClient(
        [
            "query",
            "Reasoner explanation.\nFinal Answer: A",
            "Verifier explanation.\nFinal Answer: B",
        ]
    )
    monkeypatch.setattr(client_factory, "create_llm_client", lambda model: fake_llm)
    fake_retriever = FakeRetriever(
        [Passage(passage_id="p1", source="BookA", text="relevant text", score=0.9)]
    )
    monkeypatch.setattr(
        "medqa_multiagent.entrypoint._default_retriever", lambda config: fake_retriever
    )

    outcome = logic.get_answer("Q?", {"A": "x", "B": "y"}, "V2", config, tmp_path / "cache")

    assert outcome.error is None
    assert outcome.result is not None
    assert outcome.result.variant == "V2"
    # Final answer is the Verifier's decision, not the Reasoner's raw candidate.
    assert outcome.result.answer == "B"
    assert "router_query" in outcome.result.trace
    assert "retrieved_passages" in outcome.result.trace
    assert "reasoner_candidate" in outcome.result.trace
    assert "verifier_decision" in outcome.result.trace


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


def test_load_saved_questions_returns_empty_list_when_file_is_missing(tmp_path):
    assert logic.load_saved_questions(tmp_path / "does-not-exist.jsonl") == []


def test_save_question_then_load_returns_it_back(tmp_path):
    path = tmp_path / "saved.jsonl"

    saved = logic.save_question("Q?", {"A": "x", "B": "y"}, path=path)

    loaded = logic.load_saved_questions(path)
    assert len(loaded) == 1
    assert loaded[0].question == "Q?"
    assert loaded[0].options == {"A": "x", "B": "y"}
    assert loaded[0].saved_at == saved.saved_at
    assert loaded[0].expected_answer is None
    assert loaded[0].question_id is None


def test_save_question_records_expected_answer_and_question_id_when_given(tmp_path):
    path = tmp_path / "saved.jsonl"

    logic.save_question(
        "Q?", {"A": "x"}, expected_answer="A", path=path, question_id="q123"
    )

    loaded = logic.load_saved_questions(path)
    assert loaded[0].expected_answer == "A"
    assert loaded[0].question_id == "q123"


def test_save_question_appends_rather_than_overwrites(tmp_path):
    path = tmp_path / "saved.jsonl"

    logic.save_question("Q1?", {"A": "x"}, path=path)
    logic.save_question("Q2?", {"A": "y"}, path=path)

    loaded = logic.load_saved_questions(path)
    assert [item.question for item in loaded] == ["Q1?", "Q2?"]


def test_save_question_creates_parent_directories(tmp_path):
    path = tmp_path / "nested" / "dir" / "saved.jsonl"

    logic.save_question("Q?", {"A": "x"}, path=path)

    assert path.exists()


def test_delete_saved_question_removes_only_the_selected_one(tmp_path):
    path = tmp_path / "saved.jsonl"
    logic.save_question("Q1?", {"A": "x"}, path=path)
    logic.save_question("Q2?", {"A": "y"}, path=path)
    logic.save_question("Q3?", {"A": "z"}, path=path)

    remaining = logic.delete_saved_question(1, path)

    assert [item.question for item in remaining] == ["Q1?", "Q3?"]
    assert [item.question for item in logic.load_saved_questions(path)] == ["Q1?", "Q3?"]


def test_delete_saved_question_out_of_range_raises_index_error(tmp_path):
    path = tmp_path / "saved.jsonl"
    logic.save_question("Q1?", {"A": "x"}, path=path)

    with pytest.raises(IndexError):
        logic.delete_saved_question(5, path)


def test_clear_saved_questions_empties_the_file(tmp_path):
    path = tmp_path / "saved.jsonl"
    logic.save_question("Q1?", {"A": "x"}, path=path)
    logic.save_question("Q2?", {"A": "y"}, path=path)

    logic.clear_saved_questions(path)

    assert logic.load_saved_questions(path) == []


def test_clear_saved_questions_is_a_no_op_when_nothing_was_ever_saved(tmp_path):
    path = tmp_path / "saved.jsonl"

    logic.clear_saved_questions(path)  # should not raise

    assert logic.load_saved_questions(path) == []
