import os

from medqa_multiagent.env_file import load_env_file


def test_returns_false_and_does_nothing_when_file_is_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("SOME_TEST_KEY", raising=False)

    loaded = load_env_file(tmp_path / "does-not-exist.env")

    assert loaded is False
    assert "SOME_TEST_KEY" not in os.environ


def test_loads_simple_key_value_pairs(tmp_path, monkeypatch):
    monkeypatch.delenv("SOME_TEST_KEY", raising=False)
    env_path = tmp_path / ".env"
    env_path.write_text("SOME_TEST_KEY=abc123\n", encoding="utf-8")

    loaded = load_env_file(env_path)

    assert loaded is True
    assert os.environ["SOME_TEST_KEY"] == "abc123"


def test_skips_blank_lines_and_comments(tmp_path, monkeypatch):
    monkeypatch.delenv("SOME_TEST_KEY", raising=False)
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n# a comment\nSOME_TEST_KEY=abc123\n# another comment\n\n",
        encoding="utf-8",
    )

    load_env_file(env_path)

    assert os.environ["SOME_TEST_KEY"] == "abc123"


def test_strips_leading_export_and_matching_quotes(tmp_path, monkeypatch):
    monkeypatch.delenv("SOME_TEST_KEY", raising=False)
    monkeypatch.delenv("OTHER_TEST_KEY", raising=False)
    env_path = tmp_path / ".env"
    env_path.write_text(
        'export SOME_TEST_KEY="abc123"\nOTHER_TEST_KEY=\'xyz789\'\n',
        encoding="utf-8",
    )

    load_env_file(env_path)

    assert os.environ["SOME_TEST_KEY"] == "abc123"
    assert os.environ["OTHER_TEST_KEY"] == "xyz789"


def test_does_not_override_an_already_exported_value_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv("SOME_TEST_KEY", "already-set")
    env_path = tmp_path / ".env"
    env_path.write_text("SOME_TEST_KEY=from-file\n", encoding="utf-8")

    load_env_file(env_path)

    assert os.environ["SOME_TEST_KEY"] == "already-set"


def test_override_true_overwrites_an_already_exported_value(tmp_path, monkeypatch):
    monkeypatch.setenv("SOME_TEST_KEY", "already-set")
    env_path = tmp_path / ".env"
    env_path.write_text("SOME_TEST_KEY=from-file\n", encoding="utf-8")

    load_env_file(env_path, override=True)

    assert os.environ["SOME_TEST_KEY"] == "from-file"
