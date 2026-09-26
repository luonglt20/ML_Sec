import pytest

faiss = pytest.importorskip("faiss")

from medqa_multiagent.rag.index import FaissFlatIndex  # noqa: E402


def test_build_and_search_returns_nearest_vector_first():
    vectors = [
        [1.0, 0.0],
        [0.0, 1.0],
        [0.9, 0.1],
    ]
    ids = ["p1", "p2", "p3"]
    index = FaissFlatIndex.build(vectors, ids)

    results = index.search([1.0, 0.0], top_k=2)

    assert [passage_id for passage_id, _ in results][0] == "p1"
    assert len(results) == 2


def test_search_returns_fewer_hits_than_top_k_when_index_is_smaller():
    index = FaissFlatIndex.build([[1.0, 0.0], [0.0, 1.0]], ["p1", "p2"])

    results = index.search([1.0, 0.0], top_k=5)

    assert len(results) == 2


def test_build_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        FaissFlatIndex.build([[1.0, 0.0]], ["p1", "p2"])


def test_build_rejects_empty_vectors():
    with pytest.raises(ValueError):
        FaissFlatIndex.build([], [])


def test_save_and_load_round_trips(tmp_path):
    vectors = [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]]
    ids = ["p1", "p2", "p3"]
    index = FaissFlatIndex.build(vectors, ids)
    index.save(tmp_path)

    loaded = FaissFlatIndex.load(tmp_path)
    results = loaded.search([1.0, 0.0], top_k=1)

    assert results[0][0] == "p1"


def test_load_raises_for_missing_index(tmp_path):
    with pytest.raises(FileNotFoundError):
        FaissFlatIndex.load(tmp_path / "does-not-exist")
