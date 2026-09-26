import tempfile
from pathlib import Path
from medqa_multiagent.rag.bm25 import BM25Index, tokenize
from medqa_multiagent.rag.retriever import Passage, IndexBackedRetriever
from tests.fakes import FakeEmbeddingClient, FakeVectorIndex



def test_tokenize_cleans_and_lowercases():
    assert tokenize("Hello, World!") == ["hello", "world"]
    assert tokenize("Ceftriaxone (Rocephin) 250mg.") == ["ceftriaxone", "rocephin", "250mg"]
    assert tokenize("") == []


def test_bm25_index_basic_search():
    docs = [
        ("doc1", "Ceftriaxone treats Neisseria gonorrhoeae joint infection"),
        ("doc2", "Trazodone is a sedative used for depression and insomnia"),
        ("doc3", "Pyelonephritis is a kidney infection caused by E coli"),
    ]
    index = BM25Index.build(docs)

    # Search for Ceftriaxone
    hits = index.search("ceftriaxone gonorrhoeae", top_k=2)
    assert len(hits) > 0
    assert hits[0][0] == "doc1"

    # Search for sleep
    hits_insomnia = index.search("insomnia trazodone", top_k=2)
    assert hits_insomnia[0][0] == "doc2"


def test_bm25_serialization():
    docs = [
        ("d1", "kidney failure creatinine clearance"),
        ("d2", "myocardial infarction troponin levels"),
    ]
    index = BM25Index.build(docs)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "bm25.json"
        index.save(path)
        assert path.exists()

        loaded = BM25Index.load(path)
        assert loaded.doc_ids == ["d1", "d2"]
        assert loaded.doc_lengths == {"d1": 4, "d2": 4}
        assert loaded.avg_doc_len == 4.0

        hits = loaded.search("troponin infarction", top_k=1)
        assert hits[0][0] == "d2"
