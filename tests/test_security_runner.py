import argparse
import json

from medqa_multiagent.config import RunConfig
from medqa_multiagent.data import Question
from medqa_multiagent.security.dataset import stratified_question_sample
from scripts import run_security_eval
from scripts.run_security_eval import command_summarize
from scripts.security_demo import _find_demo


def _prediction(question_id, attack, gold, predicted, target):
    return {
        "question_id": question_id,
        "model_track": "api",
        "attack_surface": "rag",
        "defense": "none",
        "attack_family": attack,
        "attack_position": "top1",
        "payload_placement": "prefix",
        "gold_answer": gold,
        "predicted_answer": predicted,
        "target_option": target,
        "is_correct": predicted == gold,
        "is_valid": predicted is not None,
        "marker_present": attack != "clean",
        "latency_seconds": 0.1,
        "prompt_tokens": 10,
        "completion_tokens": 2,
        "sanitized": False,
    }


def test_stratified_sample_is_balanced_and_reproducible():
    questions = [
        Question(f"{label}-{index}", "Q", {"A": "a", "B": "b", "C": "c", "D": "d"}, label)
        for label in "ABCD"
        for index in range(10)
    ]
    first = stratified_question_sample(questions, total=20, seed=14)
    second = stratified_question_sample(questions, total=20, seed=14)
    assert [item.question_id for item in first] == [item.question_id for item in second]
    assert {label: sum(item.answer == label for item in first) for label in "ABCD"} == {
        label: 5 for label in "ABCD"
    }


def test_summarize_command_writes_json_and_csv(tmp_path):
    predictions = tmp_path / "predictions.jsonl"
    records = [
        _prediction("q1", "clean", "A", "A", "B"),
        _prediction("q1", "combined", "A", "B", "B"),
    ]
    predictions.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    output_json = tmp_path / "summary.json"
    output_csv = tmp_path / "summary.csv"

    exit_code = command_summarize(
        argparse.Namespace(
            input=str(predictions),
            summary_json=str(output_json),
            summary_csv=str(output_csv),
        )
    )

    assert exit_code == 0
    summary = json.loads(output_json.read_text(encoding="utf-8"))
    assert len(summary["conditions"]) == 2
    assert summary["paired_comparisons"][0]["mcnemar_b"] == 1
    assert "targeted_asr" in output_csv.read_text(encoding="utf-8")


def test_summarize_command_can_join_immutable_raw_runs(tmp_path):
    clean = tmp_path / "clean.jsonl"
    attack = tmp_path / "attack.jsonl"
    clean.write_text(json.dumps(_prediction("q1", "clean", "A", "A", "B")) + "\n", encoding="utf-8")
    attack.write_text(json.dumps(_prediction("q1", "combined", "A", "B", "B")) + "\n", encoding="utf-8")
    output_json = tmp_path / "summary.json"
    output_csv = tmp_path / "summary.csv"

    assert command_summarize(
        argparse.Namespace(
            input=[str(clean), str(attack)],
            summary_json=str(output_json),
            summary_csv=str(output_csv),
        )
    ) == 0
    summary = json.loads(output_json.read_text(encoding="utf-8"))
    assert len(summary["conditions"]) == 2
    assert summary["paired_comparisons"][0]["mcnemar_b"] == 1


def test_summarize_command_deduplicates_shared_run_keys(tmp_path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    canonical = _prediction("q1", "combined", "A", "B", "B")
    canonical["run_key"] = "q1:combined-scenario"
    rerun = {**canonical, "predicted_answer": "A", "is_correct": True}
    first.write_text(json.dumps(canonical) + "\n", encoding="utf-8")
    second.write_text(json.dumps(rerun) + "\n", encoding="utf-8")
    output_json = tmp_path / "summary.json"
    output_csv = tmp_path / "summary.csv"

    assert command_summarize(
        argparse.Namespace(
            input=[str(first), str(second)],
            summary_json=str(output_json),
            summary_csv=str(output_csv),
        )
    ) == 0
    row = json.loads(output_json.read_text(encoding="utf-8"))["conditions"][0]
    assert row["n"] == 1
    assert row["correct"] == 0


def test_demo_selects_matched_local_triplet():
    common = {"question_id": "q1", "attack_surface": "rag"}
    records = [
        {**common, "model_track": "local_undefended", "attack_family": "clean"},
        {**common, "model_track": "local_undefended", "attack_family": "combined"},
        {**common, "model_track": "local_struq", "attack_family": "combined"},
    ]
    clean, attacked, defended = _find_demo(records, None)
    assert clean["attack_family"] == "clean"
    assert attacked["model_track"] == "local_undefended"
    assert defended["model_track"] == "local_struq"


def test_api_security_track_uses_uncached_client(monkeypatch, tmp_path):
    seen = []

    class StubClient:
        pass

    def fake_build(config):
        seen.append(config.model)
        return StubClient()

    monkeypatch.setattr(run_security_eval, "build_uncached_llm_client", fake_build)
    config = RunConfig.from_json_file("config.security_deepseek20.json")
    args = argparse.Namespace(track="api_heuristic_guard", cache_dir=str(tmp_path))

    _, returned_config = run_security_eval._build_eval_client(args, config)

    assert returned_config == config
    assert seen == ["deepseek-chat"]
