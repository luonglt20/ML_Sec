from argparse import Namespace

from medqa_multiagent.config import RunConfig
from scripts.run_prompt_injection import _build_retriever


def test_shared_retriever_builder_is_used_for_rag_variants(monkeypatch):
    sentinel = object()
    calls = []

    def fake_build(config, **kwargs):
        calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(
        "medqa_multiagent.rag.client_factory.build_retriever", fake_build
    )
    config = RunConfig(model="fake", temperature=0)
    args = Namespace(
        variant="all",
        rag_index_dir=None,
        embedding_cache_dir=None,
        retrieval_cache_dir=None,
    )
    assert _build_retriever(args, config) is sentinel
    assert len(calls) == 1
