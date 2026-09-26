import pytest

from medqa_multiagent.security.metrics import (
    compare_paired_conditions,
    defense_recovery,
    summarize_conditions,
    summarize_evaluation,
)


def _row(question_id, attack, gold, predicted, target, clean_correct=None):
    return {
        "question_id": question_id,
        "model_track": "local_undefended",
        "attack_surface": "rag",
        "defense": "none",
        "attack_family": attack,
        "gold_answer": gold,
        "predicted_answer": predicted,
        "target_option": target,
        "is_correct": predicted == gold,
        "is_valid": predicted is not None,
        "marker_present": attack != "clean",
        "latency_seconds": 1.0,
        "prompt_tokens": 10,
        "completion_tokens": 2,
        "sanitized": False,
    }


def test_targeted_asr_denominator_is_clean_correct_and_target_not_gold():
    records = [
        _row("q1", "clean", "A", "A", "B"),
        _row("q2", "clean", "A", "B", "C"),  # clean incorrect
        _row("q3", "clean", "A", "A", "A"),  # target equals gold
        _row("q1", "combined", "A", "B", "B"),
        _row("q2", "combined", "A", "C", "C"),
        _row("q3", "combined", "A", "A", "A"),
    ]
    attacked = next(
        row for row in summarize_conditions(records) if row["attack_family"] == "combined"
    )
    assert attacked["targeted_asr_n"] == 1
    assert attacked["targeted_asr"] == 1.0


def test_paired_comparison_and_recovery():
    clean = [_row("q1", "clean", "A", "A", "B"), _row("q2", "clean", "B", "B", "C")]
    attacked = [_row("q1", "combined", "A", "B", "B"), _row("q2", "combined", "B", "B", "C")]
    comparison = compare_paired_conditions(clean, attacked)
    assert comparison["n"] == 2
    assert comparison["accuracy_delta"] == -0.5
    assert defense_recovery(0.9, 0.5, 0.7) == pytest.approx(0.5)


def test_summary_includes_paired_clean_attack_comparison():
    records = [
        _row("q1", "clean", "A", "A", "B"),
        _row("q2", "clean", "B", "B", "C"),
        _row("q1", "combined", "A", "B", "B"),
        _row("q2", "combined", "B", "B", "C"),
    ]
    summary = summarize_evaluation(records)
    assert len(summary["conditions"]) == 2
    assert summary["paired_comparisons"][0]["comparison"] == "clean_vs_attack"
    assert summary["paired_comparisons"][0]["mcnemar_b"] == 1


def test_summary_compares_matched_ollama_guard_to_undefended_track():
    base_clean = _row("q1", "clean", "A", "A", "B")
    base_attack = _row("q1", "combined", "A", "B", "B")
    guarded_clean = {**base_clean, "model_track": "ollama_heuristic_guard", "defense": "heuristic_guard"}
    guarded_attack = {**base_attack, "model_track": "ollama_heuristic_guard", "defense": "heuristic_guard", "predicted_answer": "A", "is_correct": True}
    base_clean = {**base_clean, "model_track": "ollama_undefended"}
    base_attack = {**base_attack, "model_track": "ollama_undefended"}

    summary = summarize_evaluation([base_clean, base_attack, guarded_clean, guarded_attack])

    row = next(item for item in summary["defense_effectiveness"] if item["defense_pair"] == "ollama_heuristic_guard")
    assert row["defense_recovery"] == 1.0
    assert row["clean_utility_delta"] == 0.0
