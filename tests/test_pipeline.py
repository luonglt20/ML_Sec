from typing import Any, Dict

from medqa_multiagent.config import RunConfig
from medqa_multiagent.data import Question
from medqa_multiagent.pipeline import run_dev_evaluation
from medqa_multiagent.rag.retriever import Passage

from tests.fakes import FakeLLMClient, FakeRetriever



def make_config(**overrides: Any) -> RunConfig:
    data: Dict[str, Any] = dict(
        model="gpt-4o-mini",
        temperature=0.0,
        dev_sample_size=30,
        official_test_sample_size=20,
        seed=42,
        rag_top_k=3,
        rag_chunk_size=256,
        memory_top_k=3,
    )
    data.update(overrides)
    return RunConfig(**data)


def test_run_dev_evaluation_produces_one_record_per_question():
    questions = [
        Question(question_id="q1", question="Q1?", options={"A": "x", "B": "y"}, answer="A"),
        Question(question_id="q2", question="Q2?", options={"A": "x", "B": "y"}, answer="B"),
    ]
    client = FakeLLMClient(["Final Answer: A", "Final Answer: C"])
    config = make_config()

    records = run_dev_evaluation(questions, "V0", config, client)

    assert len(records) == 2
    assert records[0].question_id == "q1"
    assert records[0].predicted_answer == "A"
    assert records[0].correct_answer == "A"
    assert records[0].is_correct is True
    assert records[0].is_invalid is False

    assert records[1].question_id == "q2"
    assert records[1].predicted_answer == "C"
    assert records[1].correct_answer == "B"
    assert records[1].is_correct is False
    # "C" isn't a valid option for q2, but it *is* a well-formed single
    # letter -- the strict parser only judges format validity, not whether
    # the letter is a recognized option for this specific question.
    assert records[1].is_invalid is False


def test_run_dev_evaluation_marks_unparseable_response_invalid():
    questions = [Question(question_id="q1", question="Q1?", options={"A": "x"}, answer="A")]
    client = FakeLLMClient(["I really don't know."])
    config = make_config()

    records = run_dev_evaluation(questions, "V0", config, client)

    assert records[0].predicted_answer is None
    assert records[0].is_invalid is True
    assert records[0].is_correct is False


def test_run_dev_evaluation_v1_records_retrieved_passages_in_trace():
    questions = [
        Question(question_id="q1", question="Q1?", options={"A": "x", "B": "y"}, answer="A"),
    ]
    passages = [Passage(passage_id="p1", source="BookA", text="relevant text", score=0.9)]
    retriever = FakeRetriever(passages)
    client = FakeLLMClient(["Final Answer: A"])
    config = make_config()

    records = run_dev_evaluation(questions, "V1", config, client, retriever)

    assert len(records) == 1
    assert records[0].variant == "V1"
    assert records[0].trace["retrieved_passages"][0]["passage_id"] == "p1"


def test_run_dev_evaluation_v1_reuses_the_same_retriever_across_questions():
    questions = [
        Question(question_id="q1", question="Q1?", options={"A": "x"}, answer="A"),
        Question(question_id="q2", question="Q2?", options={"A": "x"}, answer="A"),
    ]
    passages = [Passage(passage_id="p1", source="BookA", text="relevant text", score=0.9)]
    retriever = FakeRetriever(passages)
    client = FakeLLMClient(["Final Answer: A", "Final Answer: A"])
    config = make_config()

    run_dev_evaluation(questions, "V1", config, client, retriever)

    assert retriever.calls == [("Q1?", config.rag_top_k), ("Q2?", config.rag_top_k)]


def test_run_dev_evaluation_v2_records_full_trace_and_verifiers_final_answer():
    questions = [
        Question(question_id="q1", question="Q1?", options={"A": "x", "B": "y"}, answer="B"),
    ]
    passages = [Passage(passage_id="p1", source="BookA", text="relevant text", score=0.9)]
    retriever = FakeRetriever(passages)
    client = FakeLLMClient(
        [
            "query",
            "Reasoner explanation.\nFinal Answer: A",
            "Verifier explanation.\nFinal Answer: B",
        ]
    )
    config = make_config()

    records = run_dev_evaluation(questions, "V2", config, client, retriever)

    assert len(records) == 1
    record = records[0]
    assert record.variant == "V2"
    # Final prediction is the Verifier's decision, not the Reasoner's
    # raw candidate ('A').
    assert record.predicted_answer == "B"
    assert record.is_correct is True
    assert record.trace["router_query"] == "query"
    assert record.trace["retrieved_passages"][0]["passage_id"] == "p1"
    assert record.trace["reasoner_candidate"]["answer"] == "A"
    assert record.trace["verifier_decision"]["decision"] == "override"
    assert record.trace["verifier_decision"]["answer"] == "B"
