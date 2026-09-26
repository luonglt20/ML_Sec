from typing import Any, Dict

from medqa_multiagent.records import (
    PredictionRecord,
    read_prediction_records,
    write_prediction_records,
)


def make_record(**overrides: Any) -> PredictionRecord:
    data: Dict[str, Any] = dict(
        question_id="dev-00000",
        variant="V0",
        predicted_answer="C",
        explanation="Because reasons.",
        correct_answer="C",
        is_correct=True,
        is_invalid=False,
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        latency_seconds=0.5,
        trace={},
    )
    data.update(overrides)
    return PredictionRecord(**data)


def test_write_and_read_round_trip(tmp_path):
    records = [
        make_record(question_id="dev-00000"),
        make_record(question_id="dev-00001", predicted_answer=None, is_correct=False, is_invalid=True),
    ]
    path = tmp_path / "predictions.jsonl"

    write_prediction_records(records, path)
    read_back = read_prediction_records(path)

    assert read_back == [record.to_dict() for record in records]


def test_write_creates_parent_directories(tmp_path):
    path = tmp_path / "nested" / "dir" / "predictions.jsonl"
    write_prediction_records([make_record()], path)
    assert path.exists()


def test_write_produces_one_json_object_per_line(tmp_path):
    path = tmp_path / "predictions.jsonl"
    write_prediction_records([make_record(), make_record(question_id="dev-00001")], path)

    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2


def test_write_overwrites_prior_contents(tmp_path):
    path = tmp_path / "predictions.jsonl"
    write_prediction_records([make_record(), make_record(question_id="dev-00001")], path)
    write_prediction_records([make_record(question_id="dev-00002")], path)

    read_back = read_prediction_records(path)
    assert len(read_back) == 1
    assert read_back[0]["question_id"] == "dev-00002"
