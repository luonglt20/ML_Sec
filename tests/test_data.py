import json

import pytest

from medqa_multiagent.data import Question, load_questions


def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record))
            fh.write("\n")


def test_load_questions_parses_valid_records(tmp_path):
    path = tmp_path / "questions.jsonl"
    _write_jsonl(
        path,
        [
            {
                "question_id": "dev-00000",
                "question": "Q1?",
                "options": {"A": "a", "B": "b"},
                "answer": "A",
            },
            {
                "question_id": "dev-00001",
                "question": "Q2?",
                "options": {"A": "c", "B": "d"},
                "answer": "B",
            },
        ],
    )

    questions = load_questions(path)

    assert questions == [
        Question(question_id="dev-00000", question="Q1?", options={"A": "a", "B": "b"}, answer="A"),
        Question(question_id="dev-00001", question="Q2?", options={"A": "c", "B": "d"}, answer="B"),
    ]


def test_load_questions_skips_blank_lines(tmp_path):
    path = tmp_path / "questions.jsonl"
    path.write_text(
        '{"question_id": "q1", "question": "Q?", "options": {"A": "a"}, "answer": "A"}\n'
        "\n"
        "   \n",
        encoding="utf-8",
    )

    questions = load_questions(path)

    assert len(questions) == 1


def test_load_questions_rejects_missing_field(tmp_path):
    path = tmp_path / "questions.jsonl"
    path.write_text(
        json.dumps({"question_id": "q1", "question": "Q?", "options": {"A": "a"}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="answer"):
        load_questions(path)


def test_load_questions_rejects_invalid_json(tmp_path):
    path = tmp_path / "questions.jsonl"
    path.write_text("not valid json", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid JSON"):
        load_questions(path)
