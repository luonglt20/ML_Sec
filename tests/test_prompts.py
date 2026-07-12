from medqa_multiagent.prompts import render_direct_prompt


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
