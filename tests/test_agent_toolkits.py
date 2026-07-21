import pytest
from medqa_multiagent.agents import (
    MemoryToolkit,
    ReasonerToolkit,
    ResearcherToolkit,
    RouterToolkit,
    VerifierToolkit,
)
from medqa_multiagent.memory import create_default_memory_store
from medqa_multiagent.rag.retriever import Passage


def test_router_toolkit():
    entities = RouterToolkit.extract_entities("Patient has Neisseria gonorrhoeae and MRSA infection.")
    assert "MRSA" in entities
    assert "neisseria" in entities

    subqueries = RouterToolkit.decompose_subqueries("Ceftriaxone for meningitis")
    assert len(subqueries) == 3
    assert "symptoms" in subqueries[0]
    assert "pathogenesis" in subqueries[1]
    assert "pharmacology" in subqueries[2]


def test_researcher_toolkit():
    passages = [
        Passage("p1", "BookA", "Ceftriaxone treats Neisseria infection.", 0.9),
        Passage("p2", "BookA", "Ceftriaxone treats Neisseria infection.", 0.88),  # Duplicate
        Passage("p3", "BookB", "Syphilis is caused by Treponema pallidum.", 0.85),
    ]

    filtered = ResearcherToolkit.mmr_filter(passages, top_k=2, mmr_lambda=0.5)
    assert len(filtered) == 2
    ids = [p.passage_id for p in filtered]
    assert "p1" in ids
    assert "p3" in ids
    assert "p2" not in ids

    covered = ResearcherToolkit.evaluate_passages_coverage("Neisseria infection", passages)
    assert covered is True


def test_memory_toolkit():
    store = create_default_memory_store()
    cases = MemoryToolkit.search_memory(store, "Gram-negative diplococci urethritis", top_k=1)
    assert len(cases) == 1
    assert cases[0].case_id == "case_001"

    pearls = MemoryToolkit.extract_pearls(cases)
    assert "Exemplar [1]" in pearls


def test_reasoner_toolkit():
    has_citations = ReasonerToolkit.verify_citations("Based on [1], Ceftriaxone is 1st line.", passage_count=1)
    assert has_citations is True

    no_citations = ReasonerToolkit.verify_citations("Ceftriaxone is 1st line.", passage_count=1)
    assert no_citations is False


def test_verifier_toolkit():
    risk = VerifierToolkit.audit_hallucination_risk(
        "The patient definitely has severe renal failure and liver cirrhosis despite no evidence provided.",
        passages=[Passage("p1", "BookA", "Text without liver details.", 0.9)],
    )
    assert risk is True

    callback_req = VerifierToolkit.issue_callback_request("Missing drug resistance info")
    assert "Callback Request" in callback_req


def test_tool_registry_function_calling_schemas():
    router_toolkit = RouterToolkit()
    schemas = router_toolkit.registry.get_schemas()
    assert len(schemas) == 6
    tool_names = [s["function"]["name"] for s in schemas]
    assert "extract_entities" in tool_names
    assert "generate_hyde_context" in tool_names
    assert "decompose_subqueries" in tool_names
    assert "prune_vignette_noise" in tool_names
    assert "fast_thinking_evaluator" in tool_names
    assert "search_strategy_planner" in tool_names

    # Test dynamic execution via registry
    res = router_toolkit.registry.execute_tool("generate_hyde_context", question="Diabetes mellitus type 2")
    assert "Diabetes mellitus type 2" in res


def test_fast_thinking_and_anti_loop_tools():
    # 1. Fast thinking evaluator
    fast_eval = RouterToolkit.fast_thinking_evaluator("What is the first-line treatment for syphilis?", {"A": "Penicillin G"})
    assert fast_eval["is_fast_path"] is True
    assert fast_eval["estimated_tokens_saved"] > 0

    # 2. Anti-Loop Circuit Breaker
    cb1 = VerifierToolkit.issue_callback_request("Missing drug sensitivity", retry_count=0)
    assert "Callback Request" in cb1

    cb2 = VerifierToolkit.issue_callback_request("Missing drug sensitivity", retry_count=2)
    assert "Max retries" in cb2


