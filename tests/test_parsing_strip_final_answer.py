from medqa_multiagent.parsing import strip_final_answer


def test_strip_final_answer_removes_the_answer_line():
    raw = "Penicillin blocks cell wall synthesis.\nFinal Answer: C"
    assert strip_final_answer(raw) == "Penicillin blocks cell wall synthesis."


def test_strip_final_answer_is_case_insensitive_and_tolerates_parens():
    raw = "Explanation text.\nfinal answer: (b)"
    assert strip_final_answer(raw) == "Explanation text."


def test_strip_final_answer_leaves_text_unchanged_when_no_match():
    raw = "No final-answer convention present here."
    assert strip_final_answer(raw) == raw


def test_strip_final_answer_trims_surrounding_whitespace():
    raw = "  Some explanation.  \nFinal Answer: A\n"
    assert strip_final_answer(raw) == "Some explanation."
