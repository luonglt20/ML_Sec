import pytest

from medqa_multiagent.rag.chunking import Chunk
from medqa_multiagent.rag.corpus import load_corpus_documents, read_chunks, write_chunks


def test_load_corpus_documents_reads_txt_files_sorted_by_filename(tmp_path):
    (tmp_path / "Zebra.txt").write_text("zebra text", encoding="utf-8")
    (tmp_path / "Anatomy.txt").write_text("anatomy text", encoding="utf-8")
    (tmp_path / "notes.md").write_text("should be ignored", encoding="utf-8")

    documents = load_corpus_documents(tmp_path)

    assert documents == [
        ("Anatomy", "anatomy text"),
        ("Zebra", "zebra text"),
    ]


def test_load_corpus_documents_raises_for_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_corpus_documents(tmp_path / "does-not-exist")


def test_write_then_read_chunks_round_trips(tmp_path):
    chunks = [
        Chunk(chunk_id="Book::chunk-0", source="Book", text="first chunk"),
        Chunk(chunk_id="Book::chunk-1", source="Book", text="second chunk"),
    ]
    path = tmp_path / "passages.jsonl"

    write_chunks(chunks, path)
    result = read_chunks(path)

    assert result == chunks


def test_write_chunks_creates_parent_directories(tmp_path):
    path = tmp_path / "nested" / "dir" / "passages.jsonl"
    write_chunks([Chunk(chunk_id="a", source="s", text="t")], path)
    assert path.exists()
