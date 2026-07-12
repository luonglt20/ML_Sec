import json

import pytest

from medqa_multiagent.config import RunConfig


def make_valid_data(**overrides):
    data = {
        "model": "gpt-4o-mini",
        "temperature": 0.0,
        "dev_sample_size": 30,
        "official_test_sample_size": 20,
        "seed": 42,
        "rag_top_k": 3,
        "rag_chunk_size": 256,
        "memory_top_k": 3,
        "rag_index_dir": "data/rag_index",
    }
    data.update(overrides)
    return data


def test_construct_valid_config():
    config = RunConfig(**make_valid_data())
    assert config.model == "gpt-4o-mini"
    assert config.temperature == 0.0
    assert config.dev_sample_size == 30
    assert config.official_test_sample_size == 20
    assert config.seed == 42
    assert config.rag_top_k == 3
    assert config.rag_chunk_size == 256
    assert config.memory_top_k == 3
    assert config.rag_index_dir == "data/rag_index"


def test_config_is_immutable():
    config = RunConfig(**make_valid_data())
    with pytest.raises(Exception):
        config.model = "deepseek-chat"  # type: ignore[misc]


@pytest.mark.parametrize(
    "overrides",
    [
        {"model": ""},
        {"model": "   "},
        {"temperature": -0.1},
        {"dev_sample_size": 0},
        {"dev_sample_size": -5},
        {"official_test_sample_size": 0},
        {"rag_top_k": 0},
        {"rag_chunk_size": 0},
        {"memory_top_k": 0},
    ],
)
def test_invalid_field_values_raise(overrides):
    with pytest.raises(ValueError):
        RunConfig(**make_valid_data(**overrides))


def test_from_mapping_round_trips_valid_data():
    data = make_valid_data()
    config = RunConfig.from_mapping(data)
    assert config.to_dict() == data


def test_from_mapping_rejects_unknown_field():
    data = make_valid_data()
    data["unexpected_field"] = 123
    with pytest.raises(ValueError, match="Unknown"):
        RunConfig.from_mapping(data)


def test_from_mapping_rejects_missing_field():
    data = make_valid_data()
    del data["seed"]
    with pytest.raises(ValueError, match="Missing"):
        RunConfig.from_mapping(data)


def test_from_json_file_round_trips(tmp_path):
    data = make_valid_data(model="deepseek-chat", seed=7)
    config_path = tmp_path / "run_config.json"
    config_path.write_text(json.dumps(data), encoding="utf-8")

    config = RunConfig.from_json_file(config_path)

    assert config.model == "deepseek-chat"
    assert config.seed == 7
    assert config.to_dict() == data


def test_from_mapping_defaults_rag_index_dir_when_omitted():
    """A config file written before `rag_index_dir` existed still loads,
    since that field declares a dataclass default."""
    data = make_valid_data()
    del data["rag_index_dir"]

    config = RunConfig.from_mapping(data)

    assert config.rag_index_dir == "data/rag_index"


def test_from_mapping_still_rejects_missing_required_field_alongside_a_defaulted_one():
    data = make_valid_data()
    del data["rag_index_dir"]
    del data["seed"]

    with pytest.raises(ValueError, match="Missing") as exc_info:
        RunConfig.from_mapping(data)

    assert "seed" in str(exc_info.value)
    assert "rag_index_dir" not in str(exc_info.value)
