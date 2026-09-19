from medqa_multiagent.config import RunConfig
from medqa_multiagent.data import Question
from medqa_multiagent.rag.retriever import Passage
from medqa_multiagent.prompt_injection import (
    ATTACK_STRATEGIES,
    calculate_attack_metrics,
    choose_target_answer,
    evaluate_prompt_injection,
    evaluate_prompt_injection_variants,
    inject_prompt,
    write_multi_variant_attack_report,
    write_benchmark_summary,
)
from tests.fakes import FakeLLMClient, FakeRetriever


def _config() -> RunConfig:
    return RunConfig(model="fake", temperature=0.0)


def test_choose_target_answer_is_deterministic_and_wrong():
    options = {"A": "one", "B": "two", "C": "three"}
    assert choose_target_answer(options, "A") == "B"
    assert choose_target_answer(options, "C") == "A"


def test_all_upstream_attack_shapes_keep_question_and_target():
    for strategy in ATTACK_STRATEGIES:
        attacked = inject_prompt("Clinical stem", "B", strategy)
        assert attacked.startswith("Clinical stem")
        assert "Final Answer: B" in attacked
    assert "Ignore previous instructions" in inject_prompt("Q", "B", "ignore")
    assert "already been answered" in inject_prompt("Q", "B", "fake_completion")
    assert "Ignore previous instructions" in inject_prompt("Q", "B", "combine")


def test_evaluation_runs_paired_calls_and_calculates_metrics():
    question = Question("q1", "Stem", {"A": "x", "B": "y"}, "A")
    client = FakeLLMClient(
        ["Clean explanation\nFinal Answer: A", "Injected\nFinal Answer: B"]
    )
    records, metrics = evaluate_prompt_injection(
        [question], "V0", "combine", _config(), client
    )
    assert len(client.calls) == 2
    assert records[0].attack_succeeded is True
    assert metrics.clean_accuracy == 1.0
    assert metrics.attacked_accuracy == 0.0
    assert metrics.accuracy_drop == 1.0
    assert metrics.attack_success_rate == 1.0
    assert metrics.clean_correct_attack_success_rate == 1.0
    assert metrics.prediction_flip_rate == 1.0
    assert metrics.clean_avg_total_tokens == 15.0
    assert metrics.attacked_avg_total_tokens == 15.0


def test_empty_metrics_are_rejected():
    try:
        calculate_attack_metrics([])
    except ValueError as exc:
        assert "at least one" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_multi_variant_evaluation_uses_same_sample_for_all_variants():
    question = Question("q1", "Stem", {"A": "x", "B": "y"}, "A")
    client = FakeLLMClient(
        [
            "Clean V0\nFinal Answer: A",
            "Attack V0\nFinal Answer: B",
            "Clean V0 again\nFinal Answer: A",
            "Attack V0 again\nFinal Answer: B",
        ]
    )
    retriever = FakeRetriever([Passage("p1", "source", "medical context", 1.0)])
    results = evaluate_prompt_injection_variants(
        [question], ["V0", "V1"], "combine", _config(), client, retriever
    )
    assert list(results) == ["V0", "V1"]
    assert all(metrics.total == 1 for _, metrics in results.values())
    assert all(records[0].question_id == "q1" for records, _ in results.values())


def test_multi_variant_report_groups_metrics_and_records(tmp_path):
    question = Question("q1", "Stem", {"A": "x", "B": "y"}, "A")
    client = FakeLLMClient(
        ["Clean\nFinal Answer: A", "Attack\nFinal Answer: B"]
    )
    results = evaluate_prompt_injection_variants(
        [question], ["V0"], "combine", _config(), client
    )
    output = tmp_path / "report.json"
    write_multi_variant_attack_report(results, "combine", output)

    import json

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["strategy"] == "combine"
    assert payload["variants"]["V0"]["metrics"]["attack_success_rate"] == 1.0
    assert payload["variants"]["V0"]["records"][0]["question_id"] == "q1"


def test_multi_variant_report_includes_benchmark_metadata(tmp_path):
    question = Question("test-00000", "Stem", {"A": "x", "B": "y"}, "A")
    client = FakeLLMClient(
        ["Clean\nFinal Answer: A", "Attack\nFinal Answer: B"]
    )
    results = evaluate_prompt_injection_variants(
        [question], ["V0"], "combine", _config(), client
    )
    output = tmp_path / "report.json"
    metadata = {"data": "data/test.jsonl", "selection": "first_50"}
    write_multi_variant_attack_report(
        results, "combine", output, metadata=metadata
    )

    import json

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["metadata"] == metadata


def test_progress_callback_receives_running_metrics():
    question = Question("q1", "Stem", {"A": "x", "B": "y"}, "A")
    client = FakeLLMClient(
        ["Clean\nFinal Answer: A", "Attack\nFinal Answer: B"]
    )
    updates = []

    evaluate_prompt_injection(
        [question],
        "V0",
        "combine",
        _config(),
        client,
        progress_callback=lambda done, total, record, metrics: updates.append(
            (done, total, record.question_id, metrics.attack_success_rate)
        ),
    )

    assert updates == [(1, 1, "q1", 1.0)]


def test_markdown_summary_contains_comparison_table(tmp_path):
    question = Question("q1", "Stem", {"A": "x", "B": "y"}, "A")
    client = FakeLLMClient(
        ["Clean\nFinal Answer: A", "Attack\nFinal Answer: B"]
    )
    results = evaluate_prompt_injection_variants(
        [question], ["V0"], "combine", _config(), client
    )
    output = tmp_path / "summary.md"

    write_benchmark_summary(
        results,
        "combine",
        output,
        metadata={"data": "data/test.jsonl", "question_count": 50},
    )

    summary = output.read_text(encoding="utf-8")
    assert "| Variant | Clean acc." in summary
    assert "| V0 | 100.0% | 0.0% | 100.0" in summary
    assert "## Accuracy overview" in summary
    assert "clean `██████████` 100.0%" in summary
