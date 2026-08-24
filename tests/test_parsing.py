import pytest

from medqa_multiagent.parsing import parse_final_answer


@pytest.mark.parametrize(
    "raw_text,expected_answer",
    [
        ("Final Answer: A", "A"),
        ("Final Answer: B.", "B"),
        ("final answer: c", "C"),
        ("FINAL ANSWER: D", "D"),
        ("Final Answer:A", "A"),
        ("Final Answer:    B", "B"),
        ("Final Answer: (C)", "C"),
        (
            "The patient's presentation is most consistent with option A.\n"
            "Final Answer: A\n"
            "This is because the symptoms align with A.",
            "A",
        ),
        ("Reasoning...\nFinal Answer: B\n", "B"),
    ],
)
def test_valid_responses_extract_expected_letter(raw_text, expected_answer):
    result = parse_final_answer(raw_text)
    assert result.is_valid is True
    assert result.answer == expected_answer
    assert result.reason is None


@pytest.mark.parametrize(
    "raw_text",
    [
        "",
        "There is no final answer line here at all.",
        "Final Answer",  # missing colon and letter
        "Final Answer A",  # missing colon
        "FinalAnswer: A",  # missing space between "Final" and "Answer"
        "Final Answer: E",  # not a valid 4-option letter
        "Final Answer: 1",  # not a letter
        "Final Answer: AB",  # not a single clean letter
        "Final Answer: A and Final Answer: B",  # ambiguous, two occurrences
        "Final Answer: A\nFinal Answer: A",  # ambiguous even if letters agree
    ],
)
def test_malformed_or_ambiguous_responses_are_invalid(raw_text):
    result = parse_final_answer(raw_text)
    assert result.is_valid is False
    assert result.answer is None
    assert result.reason is not None


def test_custom_valid_options_allows_a_fifth_letter():
    result = parse_final_answer("Final Answer: E", valid_options=("A", "B", "C", "D", "E"))
    assert result.is_valid is True
    assert result.answer == "E"


def test_custom_valid_options_still_rejects_letters_outside_the_set():
    result = parse_final_answer("Final Answer: F", valid_options=("A", "B", "C", "D", "E"))
    assert result.is_valid is False


def test_raw_text_is_preserved_on_the_result():
    raw_text = "Final Answer: A"
    result = parse_final_answer(raw_text)
    assert result.raw_text == raw_text

    raw_text_invalid = "garbage output"
    result_invalid = parse_final_answer(raw_text_invalid)
    assert result_invalid.raw_text == raw_text_invalid
