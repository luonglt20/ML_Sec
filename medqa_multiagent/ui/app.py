"""Streamlit demo UI over the black-box `answer_question` entrypoint.

A minimal single-page app for manually testing/demoing the pipeline. Lets a
user type/paste a question stem and its A-D options, pick a variant (read
dynamically from `entrypoint.SUPPORTED_VARIANTS`, so the UI needs no code
change as V1-V4 land) and a config file, then run the question through
`answer_question` and see the predicted answer, explanation, invalid-
response flag, and -- when present -- the per-agent trace.

UI-only: adds no new answer-producing logic beyond what the CLI already
calls, and has no test-set access anywhere -- only ad hoc questions typed
in by the user, or (via the optional "Load random example" button) an
example pulled from the local *dev* pool (`data/dev.jsonl`) purely as a
typing shortcut. The official test split (`data/test.jsonl`) is never
read by this UI. Reuses the exact same LLM client stack as the CLI
(`client_factory.build_llm_client`, i.e. `create_llm_client` +
`OnDiskLLMCache` + `LoggingLLMClient`), so demo runs are cached exactly like
CLI runs.

Run with:

    streamlit run medqa_multiagent/ui/app.py
"""

from __future__ import annotations

import random
from typing import Any, List

import streamlit as st

# Absolute imports (not relative) because Streamlit executes this file
# directly, rather than importing it as part of the `medqa_multiagent`
# package -- relative imports would fail with "no known parent package".
# The package must be installed (e.g. `pip install -e .`) for these to
# resolve.
from medqa_multiagent.client_factory import DEFAULT_CACHE_DIR
from medqa_multiagent.data import Question, load_questions
from medqa_multiagent.entrypoint import SUPPORTED_VARIANTS
from medqa_multiagent.env_file import load_env_file
from medqa_multiagent.ui.logic import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_SAVED_QUESTIONS_PATH,
    OPTION_LETTERS,
    SavedQuestion,
    clear_saved_questions,
    delete_saved_question,
    get_answer,
    load_config,
    load_saved_questions,
    save_question,
)

DEFAULT_ENV_FILE = ".env"

#: Deliberately the *dev* pool, never `data/test.jsonl` -- the "Load random
#: example" button is a typing shortcut for ad hoc manual testing, not a
#: sampled dev/test evaluation pass, so it must never touch the official
#: test split.
DEFAULT_EXAMPLE_DATA_PATH = "data/dev.jsonl"

_EXAMPLE_ERROR_KEY = "_example_load_error"
_EXAMPLE_QUESTION_ID_KEY = "_example_question_id"
_EXAMPLE_ANSWER_KEY = "_example_expected_answer"
_EXAMPLE_LOADED_QUESTION_KEY = "_example_loaded_question"
_EXAMPLE_LOADED_OPTIONS_KEY = "_example_loaded_options"
_EXAMPLE_SOURCE_LABEL_KEY = "_example_source_label"

_SAVED_QUESTION_SELECTBOX_KEY = "_saved_question_selectbox"


@st.cache_data(show_spinner=False)
def _load_question_pool(data_path: str) -> List[Question]:
    return load_questions(data_path)


def _load_random_example(data_path: str) -> None:
    """Populate the question/option widgets with one random dev-pool question.

    Sets `st.session_state` entries the widgets below read via their `key`s,
    so this must run *before* those widgets are instantiated in the same
    script run (the standard Streamlit pattern for programmatically setting
    a widget's value in response to a button click).
    """
    try:
        pool = _load_question_pool(data_path)
    except (OSError, ValueError) as exc:
        st.session_state[_EXAMPLE_ERROR_KEY] = f"Could not load example data from {data_path!r}: {exc}"
        return
    if not pool:
        st.session_state[_EXAMPLE_ERROR_KEY] = f"{data_path!r} has no questions to sample from."
        return

    question = random.choice(pool)
    loaded_options = {letter: question.options.get(letter, "") for letter in OPTION_LETTERS}
    st.session_state["question_stem"] = question.question
    for letter, text in loaded_options.items():
        st.session_state[f"option_{letter}"] = text

    # Remembered so the expected answer can be shown only while the
    # displayed question/options still match this loaded example -- if the
    # user edits any field afterwards, the stale expected answer is hidden
    # rather than silently pointing at a question that's no longer shown.
    st.session_state[_EXAMPLE_QUESTION_ID_KEY] = question.question_id
    st.session_state[_EXAMPLE_ANSWER_KEY] = question.answer
    st.session_state[_EXAMPLE_LOADED_QUESTION_KEY] = question.question
    st.session_state[_EXAMPLE_LOADED_OPTIONS_KEY] = loaded_options
    st.session_state[_EXAMPLE_SOURCE_LABEL_KEY] = f"the dev pool (`{data_path}`)"
    st.session_state.pop(_EXAMPLE_ERROR_KEY, None)


def _load_saved_question(saved: SavedQuestion) -> None:
    """Populate the question/option widgets with one previously saved question.

    Mirrors `_load_random_example`'s session_state-before-widget-
    instantiation pattern (see its docstring), so this must likewise run
    before the question/option widgets below are instantiated in the same
    script run. If the saved question recorded a known expected answer
    (i.e. it was saved while showing an unedited "Load random example"
    question), that expected answer is shown again exactly like a freshly
    loaded dev-pool example; otherwise any stale expected-answer bookkeeping
    from a previous load is cleared.
    """
    loaded_options = {letter: saved.options.get(letter, "") for letter in OPTION_LETTERS}
    st.session_state["question_stem"] = saved.question
    for letter, text in loaded_options.items():
        st.session_state[f"option_{letter}"] = text

    if saved.expected_answer is not None:
        st.session_state[_EXAMPLE_QUESTION_ID_KEY] = saved.question_id or "(saved question)"
        st.session_state[_EXAMPLE_ANSWER_KEY] = saved.expected_answer
        st.session_state[_EXAMPLE_LOADED_QUESTION_KEY] = saved.question
        st.session_state[_EXAMPLE_LOADED_OPTIONS_KEY] = loaded_options
        st.session_state[_EXAMPLE_SOURCE_LABEL_KEY] = "your saved questions"
    else:
        st.session_state.pop(_EXAMPLE_QUESTION_ID_KEY, None)
        st.session_state.pop(_EXAMPLE_ANSWER_KEY, None)
        st.session_state.pop(_EXAMPLE_LOADED_QUESTION_KEY, None)
        st.session_state.pop(_EXAMPLE_LOADED_OPTIONS_KEY, None)


def _saved_question_label(index: int, saved: SavedQuestion) -> str:
    """A short, unique-by-position dropdown label for one saved question.

    Prefixed with `index + 1` so labels stay unique (and thus safely
    resolvable back to their list index) even when two saved questions
    happen to share the same preview text.
    """
    first_line = saved.question.strip().splitlines()[0] if saved.question.strip() else ""
    preview = first_line if first_line else "(empty question)"
    if len(preview) > 60:
        preview = preview[:57] + "..."
    timestamp = saved.saved_at[:19].replace("T", " ")
    return f"{index + 1}. [{timestamp}] {preview}"


def _matches_loaded_example(question: str, options: dict) -> bool:
    """Whether the currently displayed question/options are still exactly
    the ones a "Load random example"/"Load selected saved question" click
    last populated (i.e. unedited since), so a known expected answer can be
    safely shown/compared.
    """
    return (
        _EXAMPLE_ANSWER_KEY in st.session_state
        and question == st.session_state.get(_EXAMPLE_LOADED_QUESTION_KEY)
        and options == st.session_state.get(_EXAMPLE_LOADED_OPTIONS_KEY)
    )


def render_trace(trace: dict) -> None:
    """Render a per-agent trace dict generically, whatever keys it contains.

    Deliberately has no per-variant/per-key special-casing -- V1-V4 can add
    whatever trace keys they like (router query, retrieved passages,
    reasoner output, verifier decision, ...) without any UI code change.
    """
    if not trace:
        st.caption("No per-agent trace for this variant.")
        return
    for key, value in trace.items():
        st.markdown(f"**{key}**")
        _render_trace_value(value)


def _render_trace_value(value: Any) -> None:
    if isinstance(value, (dict, list)):
        st.json(value)
    else:
        st.write(value)


def main() -> None:
    st.set_page_config(page_title="MedQA Multi-Agent Demo", layout="wide")
    st.title("MedQA-USMLE Multi-Agent Demo")
    st.caption(
        "Manual testing/demo UI over the black-box `answer_question` "
        "entrypoint. No official test-set access -- type or paste any ad "
        "hoc question below, or load a random dev-pool example."
    )

    with st.sidebar:
        st.header("Run configuration")
        env_file_path = st.text_input(
            "Env file path",
            value=DEFAULT_ENV_FILE,
            help=(
                "Optional .env file of KEY=VALUE provider API keys "
                "(e.g. OPENAI_API_KEY=...). Loaded before every call; "
                "values already exported in the shell always take "
                "precedence."
            ),
        )
        load_env_file(env_file_path)

        config_path = st.text_input("Config file path", value=DEFAULT_CONFIG_PATH)
        cache_dir = st.text_input("Cache directory", value=DEFAULT_CACHE_DIR)
        saved_questions_path = st.text_input(
            "Saved questions file",
            value=DEFAULT_SAVED_QUESTIONS_PATH,
            help=(
                "Local, on-disk scratchpad of questions you've chosen to "
                "keep around (via '💾 Save this question' below) for "
                "repeated manual retesting -- e.g. a question that turned "
                "out to be hard. Not read by the CLI/evaluation harness."
            ),
        )

        config, config_error = load_config(config_path)
        if config_error or config is None:
            st.error(config_error)
            st.stop()
            return

        st.text_input("Model", value=config.model, disabled=True)
        st.number_input("Temperature", value=config.temperature, disabled=True)

    if not SUPPORTED_VARIANTS:
        st.error("No variants are implemented yet.")
        st.stop()
    variant = st.selectbox("Variant", SUPPORTED_VARIANTS)

    st.subheader("Question")
    example_col, saved_col = st.columns(2)
    with example_col:
        if st.button("🎲 Load random example"):
            _load_random_example(DEFAULT_EXAMPLE_DATA_PATH)
        st.caption(
            f"Fills the fields below with a random question from "
            f"`{DEFAULT_EXAMPLE_DATA_PATH}` (the dev pool) -- a typing "
            "shortcut only, never the official test split, and not itself "
            "an evaluation run."
        )
        example_error = st.session_state.get(_EXAMPLE_ERROR_KEY)
        if example_error:
            st.warning(example_error)

    with saved_col:
        saved_questions = load_saved_questions(saved_questions_path)
        if saved_questions:
            labels = [
                _saved_question_label(index, saved) for index, saved in enumerate(saved_questions)
            ]
            selected_label = st.selectbox(
                "💾 Saved questions", labels, key=_SAVED_QUESTION_SELECTBOX_KEY
            )
            selected_index = labels.index(selected_label)
            load_col, delete_col = st.columns(2)
            with load_col:
                if st.button("📂 Load selected"):
                    _load_saved_question(saved_questions[selected_index])
                    st.rerun()
            with delete_col:
                if st.button("🗑️ Delete selected"):
                    delete_saved_question(selected_index, saved_questions_path)
                    st.rerun()
            if st.button("🧹 Clear all saved questions"):
                clear_saved_questions(saved_questions_path)
                st.rerun()
        else:
            st.caption(
                "No saved questions yet -- use \"💾 Save this question\" below "
                "to keep a hard one around for repeated retesting."
            )

    question = st.text_area("Question stem", height=150, key="question_stem")

    st.subheader("Options")
    columns = st.columns(2)
    options = {}
    for index, letter in enumerate(OPTION_LETTERS):
        with columns[index % 2]:
            options[letter] = st.text_input(f"Option {letter}", key=f"option_{letter}")

    showing_loaded_example = _matches_loaded_example(question, options)
    if showing_loaded_example:
        source_label = st.session_state.get(_EXAMPLE_SOURCE_LABEL_KEY, "the dev pool")
        st.info(
            f"Recorded expected answer for `{st.session_state[_EXAMPLE_QUESTION_ID_KEY]}` "
            f"(from {source_label}): "
            f"**{st.session_state[_EXAMPLE_ANSWER_KEY]}**"
        )

    answer_col, save_col = st.columns([3, 1])
    with answer_col:
        get_answer_clicked = st.button("Get answer", type="primary")
    with save_col:
        save_clicked = st.button("💾 Save this question")

    if save_clicked:
        if not question.strip():
            st.warning("Enter a question stem first.")
        elif not all(text.strip() for text in options.values()):
            st.warning("Fill in all four options first.")
        else:
            expected_answer = st.session_state[_EXAMPLE_ANSWER_KEY] if showing_loaded_example else None
            question_id = st.session_state[_EXAMPLE_QUESTION_ID_KEY] if showing_loaded_example else None
            save_question(
                question, options, expected_answer, saved_questions_path, question_id
            )
            st.success(
                "Saved -- reload it anytime from the \"💾 Saved questions\" "
                "selector above, even across app restarts."
            )

    if get_answer_clicked:
        if not question.strip():
            st.warning("Enter a question stem first.")
        elif not all(text.strip() for text in options.values()):
            st.warning("Fill in all four options first.")
        else:
            with st.spinner("Calling the model..."):
                outcome = get_answer(question, options, variant, config, cache_dir)

            if outcome.error is not None or outcome.result is None:
                st.error(outcome.error)
            else:
                result = outcome.result
                st.subheader("Result")
                if showing_loaded_example:
                    expected_answer = st.session_state[_EXAMPLE_ANSWER_KEY]
                    result_columns = st.columns(2)
                    result_columns[0].metric("Predicted answer", result.answer or "(none)")
                    result_columns[1].metric("Dataset expected answer", expected_answer)
                    if result.answer == expected_answer:
                        st.success("Matches the dataset's expected answer.")
                    else:
                        st.warning("Does not match the dataset's expected answer.")
                else:
                    st.metric("Predicted answer", result.answer or "(none)")
                if not result.is_valid:
                    st.warning(
                        "The model's response did not parse to exactly one "
                        "valid option letter."
                    )

                st.markdown("**Explanation**")
                st.write(result.explanation)

                with st.expander("Per-agent trace"):
                    render_trace(result.trace)

                with st.expander("Raw model response"):
                    st.text(result.raw_response)


if __name__ == "__main__":
    main()
