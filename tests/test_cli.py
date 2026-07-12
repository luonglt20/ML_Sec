import json

import pytest

from medqa_multiagent import cli, client_factory
from medqa_multiagent.rag import client_factory as rag_client_factory
from medqa_multiagent.rag.chunking import Chunk
from medqa_multiagent.rag.corpus import write_chunks
from medqa_multiagent.rag.retriever import Passage
from medqa_multiagent.records import read_prediction_records

from fakes import FakeEmbeddingClient, FakeLLMClient, FakeRetriever

pytest.importorskip("faiss")
from medqa_multiagent.rag.index import FaissFlatIndex  # noqa: E402


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
    monkeypatch.setattr(client_factory, "create_llm_client", lambda model: fake)

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


def test_answer_command_loads_the_given_env_file_before_dispatching(tmp_path, monkeypatch):
    config_path = make_config_file(tmp_path / "config.json")
    fake = FakeLLMClient(["Final Answer: A"])
    monkeypatch.setattr(client_factory, "create_llm_client", lambda model: fake)
    env_file_path = tmp_path / "custom.env"
    loaded_paths = []
    monkeypatch.setattr(cli, "load_env_file", lambda path: loaded_paths.append(path))

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
            "--env-file",
            str(env_file_path),
        ]
    )

    assert exit_code == 0
    assert loaded_paths == [str(env_file_path)]


def test_answer_command_defaults_env_file_to_dotenv(tmp_path, monkeypatch):
    config_path = make_config_file(tmp_path / "config.json")
    fake = FakeLLMClient(["Final Answer: A"])
    monkeypatch.setattr(client_factory, "create_llm_client", lambda model: fake)
    loaded_paths = []
    monkeypatch.setattr(cli, "load_env_file", lambda path: loaded_paths.append(path))

    cli.main(
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

    assert loaded_paths == [cli.DEFAULT_ENV_FILE]


def test_run_command_writes_one_record_per_sampled_question(tmp_path, monkeypatch):
    config_path = make_config_file(tmp_path / "config.json")
    data_path = make_data_file(tmp_path / "dev.jsonl")
    output_path = tmp_path / "predictions.jsonl"
    cache_dir = tmp_path / "cache"

    fake = FakeLLMClient(["Final Answer: A"] * 3)  # dev_sample_size == 3
    monkeypatch.setattr(client_factory, "create_llm_client", lambda model: fake)

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


def test_run_command_with_v1_builds_a_retriever_once_and_records_trace(tmp_path, monkeypatch):
    config_path = make_config_file(tmp_path / "config.json")
    data_path = make_data_file(tmp_path / "dev.jsonl")
    output_path = tmp_path / "predictions.jsonl"
    cache_dir = tmp_path / "cache"

    fake_client = FakeLLMClient(["Final Answer: A"] * 3)  # dev_sample_size == 3
    monkeypatch.setattr(client_factory, "create_llm_client", lambda model: fake_client)

    fake_retriever = FakeRetriever(
        [Passage(passage_id="p1", source="BookA", text="relevant text", score=0.9)]
    )
    build_calls = []

    def fake_build_retriever(config, **kwargs):
        build_calls.append(config)
        return fake_retriever

    monkeypatch.setattr(rag_client_factory, "build_retriever", fake_build_retriever)

    exit_code = cli.main(
        [
            "run",
            "--config",
            str(config_path),
            "--variant",
            "V1",
            "--data",
            str(data_path),
            "--output",
            str(output_path),
            "--cache-dir",
            str(cache_dir),
        ]
    )

    assert exit_code == 0
    # The retriever is built exactly once for the whole `run` (not once per
    # question), and reused across every question in the loop.
    assert len(build_calls) == 1
    assert len(fake_retriever.calls) == 3

    records = read_prediction_records(output_path)
    assert len(records) == 3
    for record in records:
        assert record["variant"] == "V1"
        assert record["trace"]["retrieved_passages"][0]["passage_id"] == "p1"


def test_rerunning_v1_hits_both_the_llm_and_retrieval_caches(tmp_path, monkeypatch):
    # Uses a *real* FaissFlatIndex/IndexBackedRetriever/OnDiskEmbeddingCache/
    # OnDiskRetrievalCache stack (only the MedCPT embedding model itself is
    # faked, to avoid needing real model weights) -- unlike
    # `test_run_command_with_v1_builds_a_retriever_once_and_records_trace`
    # above (which fakes `build_retriever` wholesale, bypassing its caching
    # entirely), this test needs the *actual* on-disk embedding/retrieval
    # caches to genuinely exercise re-run cache-hit behavior end to end.
    config_path = make_config_file(tmp_path / "config.json")
    data_path = make_data_file(tmp_path / "dev.jsonl")
    output_path = tmp_path / "predictions.jsonl"
    cache_dir = tmp_path / "cache"
    rag_index_dir = tmp_path / "rag_index"

    chunks = [
        Chunk(chunk_id="p1", source="BookA", text="relevant passage text"),
        Chunk(chunk_id="p2", source="BookB", text="less relevant passage text"),
    ]
    write_chunks(chunks, rag_index_dir / rag_client_factory.PASSAGES_FILENAME)
    # 1-dimensional vectors, matching `FakeEmbeddingClient`'s default,
    # length-derived fallback vector shape (see fakes.py).
    FaissFlatIndex.build([[1.0], [2.0]], ["p1", "p2"]).save(rag_index_dir)

    first_llm_client = FakeLLMClient(["Final Answer: A"] * 3)
    monkeypatch.setattr(client_factory, "create_llm_client", lambda model: first_llm_client)
    first_embedder = FakeEmbeddingClient()  # falls back to a length-derived vector
    monkeypatch.setattr(rag_client_factory, "MedCptEmbeddingClient", lambda: first_embedder)

    run_args = [
        "run",
        "--config",
        str(config_path),
        "--variant",
        "V1",
        "--data",
        str(data_path),
        "--output",
        str(output_path),
        "--cache-dir",
        str(cache_dir),
        "--rag-index-dir",
        str(rag_index_dir),
        "--embedding-cache-dir",
        str(tmp_path / "ecache"),
        "--retrieval-cache-dir",
        str(tmp_path / "rcache"),
    ]
    cli.main(run_args)
    rag_client_factory._build_retriever_cached.cache_clear()  # force reloading from disk
    first_run_records = read_prediction_records(output_path)

    # A fresh LLM client with *no* scripted responses, and a fresh embedding
    # client that would raise if actually called -- if the re-run makes any
    # real call to either, one of them fails, proving the re-run is
    # cache-hits only for both the LLM and the embedding/retrieval calls.
    second_llm_client = FakeLLMClient([])
    monkeypatch.setattr(client_factory, "create_llm_client", lambda model: second_llm_client)

    class _RaisingEmbeddingClient:
        def embed_query(self, text):
            raise AssertionError("embed_query should not be called on a cache hit")

        def embed_passages(self, texts):
            raise AssertionError("embed_passages should not be called on a cache hit")

    monkeypatch.setattr(
        rag_client_factory, "MedCptEmbeddingClient", lambda: _RaisingEmbeddingClient()
    )

    exit_code = cli.main(run_args)
    second_run_records = read_prediction_records(output_path)

    assert exit_code == 0
    assert len(second_llm_client.calls) == 0
    assert second_run_records == first_run_records


def test_rerunning_the_same_run_invocation_hits_the_cache_and_yields_identical_predictions(
    tmp_path, monkeypatch
):
    config_path = make_config_file(tmp_path / "config.json")
    data_path = make_data_file(tmp_path / "dev.jsonl")
    output_path = tmp_path / "predictions.jsonl"
    cache_dir = tmp_path / "cache"

    first_run_client = FakeLLMClient(["Final Answer: A"] * 3)
    monkeypatch.setattr(client_factory, "create_llm_client", lambda model: first_run_client)
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
    monkeypatch.setattr(client_factory, "create_llm_client", lambda model: second_run_client)
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
