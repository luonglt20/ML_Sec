from medqa_multiagent.prompts import render_direct_prompt, render_rag_prompt
from medqa_multiagent.rag.retriever import Passage


def test_render_direct_prompt_includes_question_and_options():
    prompt = render_direct_prompt(
        "What is the capital of France?",
        {"A": "Berlin", "B": "Paris", "C": "Rome", "D": "Madrid"},
    )
    assert "What is the capital of France?" in prompt
    assert "A. Berlin" in prompt
    assert "B. Paris" in prompt
    assert "C. Rome" in prompt
    assert "D. Madrid" in prompt


def test_render_direct_prompt_instructs_final_answer_format():
    prompt = render_direct_prompt("Q?", {"A": "x", "B": "y"})
    assert "Final Answer: <letter>" in prompt


def test_render_direct_prompt_lists_options_in_sorted_order():
    prompt = render_direct_prompt("Q?", {"C": "third", "A": "first", "B": "second"})
    first_index = prompt.index("A. first")
    second_index = prompt.index("B. second")
    third_index = prompt.index("C. third")
    assert first_index < second_index < third_index


def test_render_direct_prompt_strips_surrounding_whitespace_from_question():
    prompt = render_direct_prompt("  Q?  \n", {"A": "x"})
    assert "  Q?  " not in prompt
    assert "Q?" in prompt


def test_render_direct_prompt_is_a_pure_function_of_its_arguments():
    prompt_a = render_direct_prompt("Q?", {"A": "x", "B": "y"})
    prompt_b = render_direct_prompt("Q?", {"A": "x", "B": "y"})
    assert prompt_a == prompt_b


def _make_passages():
    return [
        Passage(passage_id="p1", source="BookA", text="Penicillin inhibits cell wall synthesis.", score=0.9),
        Passage(passage_id="p2", source="BookB", text="Beta-lactams target transpeptidases.", score=0.7),
    ]


def test_render_rag_prompt_includes_question_and_options():
    prompt = render_rag_prompt(
        "What is the capital of France?",
        {"A": "Berlin", "B": "Paris", "C": "Rome", "D": "Madrid"},
        [],
    )
    assert "What is the capital of France?" in prompt
    assert "A. Berlin" in prompt
    assert "B. Paris" in prompt


def test_render_rag_prompt_includes_retrieved_passage_text_and_source():
    prompt = render_rag_prompt("Q?", {"A": "x", "B": "y"}, _make_passages())
    assert "Penicillin inhibits cell wall synthesis." in prompt
    assert "Beta-lactams target transpeptidases." in prompt
    assert "BookA" in prompt
    assert "BookB" in prompt


def test_render_rag_prompt_lists_passages_in_given_order_without_resorting():
    passages = _make_passages()
    prompt = render_rag_prompt("Q?", {"A": "x"}, passages)
    first_index = prompt.index(passages[0].text)
    second_index = prompt.index(passages[1].text)
    assert first_index < second_index


def test_render_rag_prompt_with_no_passages_says_so_explicitly():
    prompt = render_rag_prompt("Q?", {"A": "x"}, [])
    assert "(none retrieved)" in prompt


def test_render_rag_prompt_instructs_final_answer_format():
    prompt = render_rag_prompt("Q?", {"A": "x", "B": "y"}, _make_passages())
    assert "Final Answer: <letter>" in prompt


def test_render_rag_prompt_is_a_pure_function_of_its_arguments():
    prompt_a = render_rag_prompt("Q?", {"A": "x"}, _make_passages())
    prompt_b = render_rag_prompt("Q?", {"A": "x"}, _make_passages())
    assert prompt_a == prompt_b
