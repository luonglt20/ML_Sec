import pytest

pytest.importorskip("faiss")

from medqa_multiagent.config import RunConfig
from medqa_multiagent.rag import client_factory as rag_client_factory
from medqa_multiagent.rag.chunking import Chunk
from medqa_multiagent.rag.corpus import write_chunks
from medqa_multiagent.rag.index import FaissFlatIndex

from tests.fakes import FakeEmbeddingClient



def make_config(**overrides):
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
    return RunConfig.from_mapping(data)


def build_fake_index_dir(directory):
    chunks = [
        Chunk(chunk_id="p1", source="BookA", text="alpha passage"),
        Chunk(chunk_id="p2", source="BookB", text="beta passage"),
    ]
    write_chunks(chunks, directory / rag_client_factory.PASSAGES_FILENAME)
    index = FaissFlatIndex.build([[1.0, 0.0], [0.0, 1.0]], ["p1", "p2"])
    index.save(directory)
    return directory


@pytest.fixture(autouse=True)
def _clear_retriever_cache():
    # `_build_retriever_cached` is a process-wide `functools.lru_cache`;
    # clear it before/after every test so tests don't leak memoized
    # retrievers into each other via a re-used tmp_path string.
    rag_client_factory._build_retriever_cached.cache_clear()
    yield
    rag_client_factory._build_retriever_cached.cache_clear()


def test_build_retriever_raises_when_index_directory_is_missing(tmp_path):
    config = make_config(rag_index_dir=str(tmp_path / "missing"))

    with pytest.raises(FileNotFoundError):
        rag_client_factory.build_retriever(config)


def test_build_retriever_loads_index_and_retrieves(tmp_path, monkeypatch):
    index_dir = build_fake_index_dir(tmp_path / "index")
    fake_embedder = FakeEmbeddingClient({"query text": [1.0, 0.0]})
    monkeypatch.setattr(rag_client_factory, "MedCptEmbeddingClient", lambda: fake_embedder)
    config = make_config(rag_index_dir=str(index_dir))

    retriever = rag_client_factory.build_retriever(
        config,
        embedding_cache_dir=str(tmp_path / "ecache"),
        retrieval_cache_dir=str(tmp_path / "rcache"),
    )
    passages = retriever.retrieve("query text", top_k=1)

    assert passages[0].passage_id == "p1"
    assert passages[0].source == "BookA"
    assert passages[0].text == "alpha passage"


def test_build_retriever_reuses_the_same_in_memory_retriever(tmp_path, monkeypatch):
    index_dir = build_fake_index_dir(tmp_path / "index")
    construction_count = {"n": 0}

    def fake_constructor():
        construction_count["n"] += 1
        return FakeEmbeddingClient({"query text": [1.0, 0.0]})

    monkeypatch.setattr(rag_client_factory, "MedCptEmbeddingClient", fake_constructor)
    config = make_config(rag_index_dir=str(index_dir))
    embedding_cache_dir = str(tmp_path / "ecache")
    retrieval_cache_dir = str(tmp_path / "rcache")

    first = rag_client_factory.build_retriever(
        config, embedding_cache_dir=embedding_cache_dir, retrieval_cache_dir=retrieval_cache_dir
    )
    second = rag_client_factory.build_retriever(
        config, embedding_cache_dir=embedding_cache_dir, retrieval_cache_dir=retrieval_cache_dir
    )

    assert first is second
    assert construction_count["n"] == 1


def test_build_retriever_index_dir_argument_overrides_config(tmp_path, monkeypatch):
    index_dir = build_fake_index_dir(tmp_path / "explicit_index")
    monkeypatch.setattr(
        rag_client_factory,
        "MedCptEmbeddingClient",
        lambda: FakeEmbeddingClient({"query text": [1.0, 0.0]}),
    )
    config = make_config(rag_index_dir=str(tmp_path / "not_used"))

    retriever = rag_client_factory.build_retriever(
        config,
        index_dir=index_dir,
        embedding_cache_dir=str(tmp_path / "ecache"),
        retrieval_cache_dir=str(tmp_path / "rcache"),
    )
    passages = retriever.retrieve("query text", top_k=1)

    assert passages[0].passage_id == "p1"
