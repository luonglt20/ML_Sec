import json

from medqa_multiagent import cli
from medqa_multiagent.records import read_prediction_records

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


def make_data_file(path):
    questions = [
        {"question_id": f"dev-{i:05d}", "question": f"Question {i}?", "options": {"A": "x", "B": "y"}, "answer": "A"}
        for i in range(5)
    ]
    with open(path, "w", encoding="utf-8") as fh:
        for question in questions:
            fh.write(json.dumps(question))
            fh.write("\n")
    return path


def test_answer_command_prints_answer_and_explanation(tmp_path, monkeypatch, capsys):
    config_path = make_config_file(tmp_path / "config.json")
    fake = FakeLLMClient(["Some reasoning.\nFinal Answer: B"])
    monkeypatch.setattr(cli, "create_llm_client", lambda model: fake)

    exit_code = cli.main(
        [
            "answer",
            "--config",
            str(config_path),
            "--question",
            "Q?",
            "--options",
            json.dumps({"A": "x", "B": "y"}),
            "--cache-dir",
            str(tmp_path / "cache"),
        ]
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output == {"answer": "B", "explanation": "Some reasoning."}


def test_run_command_writes_one_record_per_sampled_question(tmp_path, monkeypatch):
    config_path = make_config_file(tmp_path / "config.json")
    data_path = make_data_file(tmp_path / "dev.jsonl")
    output_path = tmp_path / "predictions.jsonl"
    cache_dir = tmp_path / "cache"

    fake = FakeLLMClient(["Final Answer: A"] * 3)  # dev_sample_size == 3
    monkeypatch.setattr(cli, "create_llm_client", lambda model: fake)

    exit_code = cli.main(
        [
            "run",
            "--config",
            str(config_path),
            "--data",
            str(data_path),
            "--output",
            str(output_path),
            "--cache-dir",
            str(cache_dir),
        ]
    )

    assert exit_code == 0
    records = read_prediction_records(output_path)
    assert len(records) == 3
    for record in records:
        assert record["variant"] == "V0"
        assert record["predicted_answer"] == "A"
        assert record["is_correct"] is True
        assert record["is_invalid"] is False


def test_rerunning_the_same_run_invocation_hits_the_cache_and_yields_identical_predictions(
    tmp_path, monkeypatch
):
    config_path = make_config_file(tmp_path / "config.json")
    data_path = make_data_file(tmp_path / "dev.jsonl")
    output_path = tmp_path / "predictions.jsonl"
    cache_dir = tmp_path / "cache"

    first_run_client = FakeLLMClient(["Final Answer: A"] * 3)
    monkeypatch.setattr(cli, "create_llm_client", lambda model: first_run_client)
    cli.main(
        [
            "run",
            "--config",
            str(config_path),
            "--data",
            str(data_path),
            "--output",
            str(output_path),
            "--cache-dir",
            str(cache_dir),
        ]
    )
    first_run_records = read_prediction_records(output_path)

    # A fresh client with *no* scripted responses -- if the re-run makes any
    # real call, FakeLLMClient raises, proving the re-run is cache-hits only.
    second_run_client = FakeLLMClient([])
    monkeypatch.setattr(cli, "create_llm_client", lambda model: second_run_client)
    exit_code = cli.main(
        [
            "run",
            "--config",
            str(config_path),
            "--data",
            str(data_path),
            "--output",
            str(output_path),
            "--cache-dir",
            str(cache_dir),
        ]
    )
    second_run_records = read_prediction_records(output_path)

    assert exit_code == 0
    assert len(second_run_client.calls) == 0
    assert second_run_records == first_run_records
